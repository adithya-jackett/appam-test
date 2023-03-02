"""
General Purpose helpers and Config for pipelines etc
"""
import os
import re
import requests
import json
from pdf2image import convert_from_bytes
from typing import Dict, Union

from PIL import Image
import cv2

import multiprocessing
import shutil

import zipfile
import streamlit as st
from PIL import Image
import numpy as np

from pymongo import MongoClient, DESCENDING
import certifi
from datetime import datetime
import uuid
import base64
from io import BytesIO
import imagesize

import streamlit.components.v1 as st_components
import pandas as pd
import psutil
from bs4 import BeautifulSoup


ENV = os.getenv("ENV","PROD")

MATHPIX_API_ID = st.secrets["MATHPIX_API_ID"]
MATHPIX_API_KEY = st.secrets["MATHPIX_API_KEY"]
GOOGLE_CREDENTIALS_DATA = st.secrets["GOOGLE_CREDENTIALS_DATA"]


loginUrl = st.secrets[ENV]["loginUrl"]
ADD_QUESTION_URL = st.secrets[ENV]["ADD_QUESTION_URL"]
IMAGE_DETECTION_URL = st.secrets[ENV]["IMAGE_DETECTION_URL"]
SUGGESTED_LABELLING_URL = st.secrets[ENV]["SUGGESTED_LABELLING_URL"]
SUGGESTED_LABELLING_URL_V2 = st.secrets[ENV]["SUGGESTED_LABELLING_URL_V2"]
UPLOAD_IMAGE_URL = st.secrets[ENV]["UPLOAD_IMAGE_URL"]
BBOX_MONGO_URL = st.secrets[ENV]["BBOX_MONGO_URL"]
ALLOTMENT_SHEET_ID = st.secrets[ENV]["ALLOTMENT_SHEET_ID"]
ALLOTMENT_WORKSHEET_NAME = st.secrets[ENV]["ALLOTMENT_WORKSHEET_NAME"]
TEACHER_NAME_SHEET_ID = st.secrets[ENV]["TEACHER_NAME_SHEET_ID"]
TEACHER_NAME_WORKSHEET = st.secrets[ENV]["TEACHER_NAME_WORKSHEET"]
JACKETT_WEBSERVICE_URL = st.secrets[ENV]["JACKETT_WEBSERVICE_URL"]

BBOX_DB = MongoClient(BBOX_MONGO_URL, tlsCAFile=certifi.where()).test["bounding_box"]


MAX_THREADS = min(4, multiprocessing.cpu_count())
basewidth = 1100 # image resizing 

MAPPING = {
    'Subject':"subjectTags",
    "Classes": "classes",
    "Chapter": "chapter",
    "Curriculum":"curriculum",
    "Source": "sources",
    "Topic": 'topic',
}

def override_print_with_date():
    '''
    Override Print statement to include the Time for the logging by default
    '''
    import pytz

    IST = pytz.timezone('Asia/Kolkata')

    _print = print # keep a local copy of the original print
    return lambda *args, **kwargs: _print(datetime.now(IST).strftime("%D %H:%M:%S"), *args, **kwargs)

print_with_date = override_print_with_date()


def connect_DB(USERNAME, PASSWORD):

    login_headers = {"accept": "application/json",
                    "content-type": "application/json"
                    }

    login_data = {"username":USERNAME,
            "password":PASSWORD}
        
    respLogin = requests.post(url=loginUrl, headers=(login_headers), data=json.dumps(login_data))

    respLoginJson = respLogin.json()


    TOKEN = respLoginJson['data']['token']
    USERNAME = respLoginJson['data']['username']

    return TOKEN, USERNAME



def extract_images_from_pdf(PDF_FILE, destination_path, first_page = None, last_page = None):
    
    try:
        os.mkdir(destination_path)
    except:
        shutil.rmtree(destination_path)
        os.mkdir(destination_path)


    images = convert_from_bytes(PDF_FILE, 250, destination_path, fmt = "jpeg", output_file = "thread",
        first_page = first_page, last_page = last_page, paths_only = True, thread_count = MAX_THREADS)

    return images


def get_mathPix_OCR(image_path:str, MATHPIX_API_ID:str, MATHPIX_API_KEY:str):

    mathPix_raw_out = requests.post("https://api.mathpix.com/v3/text",
                            files={"file": open(image_path,"rb")},
                            data={
                                    "options_json":json.dumps({
                                        "formats":['html'],  # data , need "text" ?
                                        "include_line_data": True,
                                        "enable_spell_check": True,
                                        "include_smiles": True,
                                        
                                        "data_options": {"include_mathml": True, 'include_table_html':True}, # include_latex and others

                                    })
                                },
                            headers={
                                    "app_id": MATHPIX_API_ID,
                                    "app_key": MATHPIX_API_KEY
                                }
                             )

    mathPix_json_out = mathPix_raw_out.json()     
    return mathPix_json_out


def get_suggested_tagsV2(questionText:str)-> Union[Dict, str]:
    try:
        questionText = str(questionText)
        
        sl_data = {}
        sl_data['question'] = questionText

        suggested_labelling_resp = requests.put(SUGGESTED_LABELLING_URL_V2, json=sl_data)
        suggested_labelling_resp_json = suggested_labelling_resp.json()
            
        recommended_tags = suggested_labelling_resp_json["data"]["recommendedTagsV2"]

        return recommended_tags, suggested_labelling_resp_json['data']['suggestedLabellingId']
    except Exception as e:
        print("get_suggested_tagsV2 Failed. Setting Labelling Id and Tags to None")
        return None, None


def callAddQuestionAPI(questionTemplateJackettWebservice, ADD_QUESTION_URL, headers, username, author, questionText, options, answers, tags, boundingBoxId, suggestedLabellingId, sourceImageId):
    questionTemplateJackettWebservice['username'] = username
    questionTemplateJackettWebservice['author'] = author
    questionTemplateJackettWebservice['questionText'] = questionText
    questionTemplateJackettWebservice['answers'] = answers
    questionTemplateJackettWebservice['tag'] = tags
    questionTemplateJackettWebservice['options'] = options
    questionTemplateJackettWebservice['boundingBoxId'] = boundingBoxId
    questionTemplateJackettWebservice['suggestedLabellingId'] = suggestedLabellingId
    questionTemplateJackettWebservice['sourceImageId'] = sourceImageId     
    
    payload = json.dumps(questionTemplateJackettWebservice)
    return requests.post(ADD_QUESTION_URL, headers=headers, data=payload).json()


def is_image(image_path):
    return image_path.split('.')[-1] in ["jpg","JPG", "PNG", "png", 'JPEG', "jpeg"]


def image_name_sorting(image_name):
    '''
    Criteria for sorting the images name based on some number. Used as a `key` argument in sorted()
    '''
    split_num = None

    if "-<SPLIT-" in image_name:
        image_name, split_num = image_name.split("-<SPLIT-")

    found = re.findall("\d+",image_name)
    if not found : raise Exception("No number found for sorting criteria. Rename you images on some numbering basis like page or so")

    return int(found[-1] + split_num[0]) if split_num else int(found[-1])


def insert_image_bb_data(sourceImageId, image_name, all_bboxes, header, layout_type):
    '''
    Insert a New Document in the Bounding Box Collection
    '''
    header_margin = header if header else 0

    pred_coors = []
    for split_num in all_bboxes.keys():
        for coors, bbox_id in all_bboxes[split_num]:
            if coors is None:
                print_with_date("None in the Bounding box means there's no BB so not pushing to DB")
                return None
            
            pred_coors.append({
                        "boundingBoxId": bbox_id,
                        "xmin": int(coors[0]),
                        "ymin": int(coors[1]) + header_margin,
                        "xmax": int(coors[2]),
                        "ymax": int(coors[3]) + header_margin,
                        "status": "UNCHANGED",
                        "label": "LABEL_0"})
    
    insertion_doc = {
            'predictionId': str(uuid.uuid4()),
            'imageId':sourceImageId,
            'modelCard':None,
            'createdAt':datetime.now(),
            'imageKey':sourceImageId,
            'filename': image_name,
            'layoutType': str(layout_type),
            'updatedCoordinates': pred_coors,
            'predictedCoordinates': pred_coors}

    BBOX_DB.insert_one(insertion_doc)


def update_existing_bb(sourceImageId, bbox_id, coors):
    '''
   Update a new Question Document in the Bounding Box Collection
    '''
    to_push = {
                "boundingBoxId": bbox_id,
                "xmin": coors[0],
                "xmax": coors[2],
                "ymin": coors[1],
                "ymax": coors[3],
                "status": "UNCHANGED",
                "label": "LABEL_0"
            }
    BBOX_DB.update_one({'imageId': sourceImageId}, {'$push': {'predictedCoordinates': to_push}})


def get_layout_type(cols, q_type):
    '''
    Get the layout type given the No of Cols and Type of Question
    '''
    if q_type == "MCQ" and cols == 1:
        return "layout_1"
    
    if q_type == "MCQ" and cols > 1:
        return "layout_2"
    
    if q_type == "OPEN_SHORT" and cols == 1:
        return "layout_3"
    
    if q_type == "OPEN_SHORT" and cols > 1:
        return "layout_4"


def cleanText(html:str) -> str:
    '''
    Clean HTML string by removing [div, span, mathml] -> Convert every <REPLACE> tag to it's original form <li> -> remove (.: ) from left 
    '''
    if not html: return ""
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup.find_all(["div", "span", "mathml"]): # removes all div, span and mathml tags only. Keeps the content
        tag.unwrap()
    
    for tag in soup.select('REPLACEli'): # change every <REPLACE> tag to <li>
        tag.name = 'li'
    
    return str(soup).lstrip("\n .:") # test if using \s+ is necessary as in HTML it doesn't matter and in text it was there originally


def preprocess_text(html:str, selected_regex:str = None) ->str:
    '''
    Change Every NESTED <li> tag (whose ANY parent is <li>)  OR  any <li> tag whose parent is not <ol>
    AND
    `<ol start = "NUM"> <li> some data </li>` to -> 'NUM.  some data'  if selected_regex == 'q3'
    '''
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup.find_all('li', recursive=True): # change child <li> to <REPLACEli>
        if (tag.find_parents("li") or (not tag.find_parents("ol"))): tag.name = 'REPLACEli'
    
    if selected_regex == 'q3': # if selected_regex == q3, try to force change every <li> elemment in a number format
        for ol in soup.find_all("ol", start = True):
            start = int(ol["start"])
            for li in ol.find_all("li"):
                li.insert_before("\n"+str(start)+".\n")
                li.unwrap()
                start += 1
            ol.unwrap()
    
    return str(soup).replace("<br/>","<br>") # because MCQ won't be selected or we'll have to change MCQ finder Regex


def post_single_image_db(header, image)->str:
    '''
    Post any image to Backend and get image key
    '''
    if isinstance(image, str):
        ext = image.split(".")[-1].lower()
        with open(image, "rb") as f:im_bytes = f.read()
    else:
        ext = "png"
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        im_bytes = buffer.getvalue()    

    image_b64 = base64.b64encode(im_bytes).decode("utf8")
    payload = json.dumps({'fileData':image_b64,'contentType': f"image/{ext}"})
    return requests.post(UPLOAD_IMAGE_URL,  headers=header, data = payload).json()['data']['fileKey'] # sent from backend after saving image



def resize(image, new_width_height = 1920):
    '''
    Resize and return Given Image
    args:
    path: Image Path
    new_width_height = Reshaped image's width and height. If integer is given, it'll keep the aspect ratio as it is by shrinking the Bigger dimension (width or height) to the max of new_width_height  and then shring the smaller dimension accordingly 
    '''
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)

    w, h = image.size

    if (w > new_width_height) or (h > new_width_height):

        fixed_size = new_width_height if isinstance(new_width_height, int) else False

        if fixed_size:
            if h > w:
                fixed_height = fixed_size
                height_percent = (fixed_height / float(h))
                width_size = int((float(w) * float(height_percent)))
                image = image.resize((width_size, fixed_height), Image.Resampling.NEAREST)

            else:
                fixed_width = fixed_size
                width_percent = (fixed_width / float(w))
                height_size = int((float(h) * float(width_percent)))
                image = image.resize((fixed_width, height_size), Image.Resampling.NEAREST) 

        else:
            image = image.resize(new_width_height)

    return image


def handle_diagrams(diag_bboxes:list, original_image_name:str, q_type:str, post_header:dict, num_options:int = None, header_margin:int=0):
    '''
    Crop the original image based on BBoxes of diagrams and send it to Backend to get Image ids. These ids will be added to the Questions ans options part
    '''
    length = 0 if not diag_bboxes else len(diag_bboxes)
    question_part = []
    option_part = []
    if not length: return question_part, option_part

    image = Image.open(original_image_name)

    list_ids = []
    for bbox in diag_bboxes: # crop each image and push to the DB to get object_id
        bbox[1] += header_margin
        bbox[3] += header_margin

        crop = image.crop(bbox)
        try:
            list_ids.append(post_single_image_db(post_header, crop))
        except Exception as e:
            print_with_date(f"Problem in pushing crop id to backend: {e} for image name {original_image_name}")
            list_ids.append(None)
            
    if q_type == 'MCQ':
        if length < 4: # it's part of question:
            question_part = list_ids

        elif length == 4: # all in options
            option_part = list_ids
        
        else: # last 4 in options and remaining in question
            question_part = list_ids[:-num_options] # question
            option_part = list_ids[-num_options:] # last N

    else: # all in question or maybe Answer but rarely
        question_part = list_ids
    
    return question_part, option_part


def resize_all_images(input_images):
    '''
    Resize all images as bigger size images are not supported in the backend as well as Mathpix also
    '''
    for image_path in input_images:

        width, height = imagesize.get(image_path)
        if (width < 1920) and (height < 1920): continue
        
        if image_path.lower().endswith(".png"):
            resize(image_path).save(image_path)
        else:
            resize(image_path).save(image_path, quality=100, subsampling=0) # to preserve the quality


def automatic_download(object_to_download, download_filename):
    """
    Downloads a file automatically without "Download" Button.
    args:
        object_to_download:  The object to be downloaded.
        download_filename (str): filename and extension of file. e.g. mydata.csv,
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    download_script = f"""
    <html>
    <head>
    <title>Start Auto Download file</title>
    <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script>
    $('<a href="data:text/csv;base64,{b64}" download="{download_filename}">')[0].click()
    </script>
    </head>
    </html>
    """
    st_components.html(download_script)


def process_chapter_topic_sheet(create:str = "chapter"):
    '''
    Uploading a chapter - Topic sheet, return the dict values for page number limit
    args:
        create: Whether to create 'chapter' or 'topic'
    '''
    if st.session_state["chapter_sheet"] is not None:
        if not isinstance(st.session_state["chapter_sheet"], pd.DataFrame): # it gets read only once
            chap_file = st.session_state["chapter_sheet"]
            if chap_file.name.endswith("csv"): df = pd.read_csv(chap_file, header=None)
            else: df = pd.read_excel(chap_file, header=None)
            st.session_state["chapter_sheet"] = df 

        else: df = st.session_state["chapter_sheet"]

        df = df.dropna(axis=1, how = 'all') # drop extra EMPTY columns in case they were there 
        df.columns = range(df.shape[1]) # Rename COLUMN NAMES as [0,1,2,3] as [chapter -> PAGE_NUM, topic -> PAGE_NUM]

        if create == 'chapter': df = df.iloc[:,[0,1]]
        else:
            if df.shape[1] <= 2: return None, None # No Topic Present
            df = df.iloc[:,[2,3]]
        
        df = df.dropna().reset_index(drop = True) # drop empty rows
        df.columns = range(df.shape[1]) # Rename COLUMN NAMES as [0,1] where 0 pointing to page_namme. 1 pointing to Page Number
        if df.shape[0] == 1: df.loc[len(df.index)] = ['DUMMY CHAPTER', 99999] # creating problem when Only 1 chapter is used
        df = df.sort_values(by = 1, ascending=True) # index 0 is chapter name, 1 is the limit it ends at
        
        page_pointer = df.iloc[0,1]  #  starting limit of page num. Any page number below this, points to this key
        df.iat[-1,1] = 99999 # last one is the end so put the max value in case it was missed
        return dict(zip(pd.to_numeric(df[1], downcast='integer'), df[0].values)), int(page_pointer)


def show_system_stat(ram_only = True, full = False):
    '''
    Print and return System Stats
    '''
    ram_stat = psutil.virtual_memory() if full else psutil.virtual_memory().percent
    print_with_date('Memory stats:', ram_stat)
    st.write('RAM Memory stats:', ram_stat)

    if ram_only: return True

    curr_dir = os.getcwd()
    hard_disk = psutil.disk_usage(curr_dir) if full else psutil.disk_usage(curr_dir).percent

    print_with_date(f"Disk usage stats of {curr_dir}: {hard_disk}\n")
    st.write(f"Hard Disk usage stats: {hard_disk}")


def _detect_columns(path, k, block_width_thresh = 0.20):

    img = cv2.imread(path, 0)
    h, w = img.shape
    img = img[int(h*0.3):(h - int(h*0.3)), :] # Keep 40% Horizontal middle part only and 90% Vertical

    blur = cv2.GaussianBlur(img,(7,7),0)

    ret,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,k))  # to manipulate the orientation of dilution , large X means horizonatally effect more, large Y means vertically dilating more
    mask = cv2.erode(mask, kernel, iterations=10)  # erode -> Increase size of black , more the iteration more erosion

    mask_col_means =  mask.mean(axis = 0).astype(int) # mean of each pixel column
    prev = mask_col_means[0] # Last accessed Pixel column

    result = {} # stores [start_index, end_index, length_of block] for each White (background) and black (words) found

    if prev not in result:
        result[prev] = []

    result[prev].append([0]) # first column could be Black or White. Mean is other than 0 or 255 when there are mixed pixels

    first_word_pixel_index = None # When does first column comes which is not white, it is the place where ANY of the word came
    for ind, curr_col_mean in enumerate(mask_col_means[1:]): # index starts from 0 but it'll already be at Col Num 1 so add +1 to evert index number
        if curr_col_mean not in result: result[curr_col_mean] = []
        
        index = ind + 1

        if first_word_pixel_index is None and curr_col_mean != 255: first_word_pixel_index = index
        if first_word_pixel_index and curr_col_mean != 255: last_word_pixel_index = index # what is the last place where ANY Non, white column occured

        if prev != curr_col_mean: # if different block starts

            block_length = index-1  - result[prev][-1][0] # length of the block that just ended at last index

            result[prev][-1].extend([index-1, block_length]) # last block was the ending index of last block
            result[curr_col_mean].append([index]) # current index is the starting point of this block

        prev = curr_col_mean

    block_length = index-1  - result[prev][-1][0]
    result[prev][-1].extend([index-1, block_length])

    width_text_area = last_word_pixel_index - first_word_pixel_index

    num_cols = 0
    for black_col in result[0]:
        if black_col[-1] >= (block_width_thresh * width_text_area): num_cols += 1

    return num_cols


def get_cols_data(image_paths):
    col_data = {1:0, 2:0, 3:0, "Error": 0}

    for index, image in enumerate(image_paths):
        try:
            num_cols = _detect_columns(image, k = 91, block_width_thresh = 0.23)
            col_data[num_cols] += 1

        except:
            col_data["Error"] += 1
    
    return col_data