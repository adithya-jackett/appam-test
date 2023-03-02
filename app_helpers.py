"""
Helpers for Streamlit app relating to states, rendering etc
"""
from google_sheet_helpers import get_digitiser_name_tuple, get_export_worksheet_names, get_teacher_name_tuple, get_teacher_name_worksheet_url, get_allotment_worksheet_url, update_export_sheet
from helpers import *
import plotly.express as px
from mixed_pipeline import run_mixed_pipeline
from math import modf
from google_sheet_helpers import sh
from traceback import print_exc


def init_states():
    '''
    Init new states
    '''
    if 'show_second_flow' not in st.session_state:
        st.session_state['show_second_flow'] = False

    if "zipped_path" not in st.session_state:
        st.session_state['zipped_path'] = None

    if "display_image" not in st.session_state:
        st.session_state['display_image'] = None

    if "image_points" not in st.session_state: # Points to plot on image [header margin ,footer margin , vert_line-1, vert_line-1]
        st.session_state['image_points'] = [30,30,None,None]

    if 'image_file_paths' not in st.session_state:
        st.session_state['image_file_paths'] = None
    
    if 'image_splits_mapping' not in st.session_state:
        st.session_state['image_splits_mapping'] = {}

    if "file_dir" not in st.session_state:
        st.session_state['file_dir'] = None
    
    if "image_counter" not in st.session_state:
        st.session_state['image_counter'] = 0

    if "Line 1" not in st.session_state:
        st.session_state["Line 1"] = 0

    if "Line 2" not in st.session_state:
        st.session_state["Line 2"] = 0
    
    if "col_detection" not in st.session_state:
        st.session_state["col_detection"] = None
    
    if "chapter_sheet" not in st.session_state:
        st.session_state["chapter_sheet"] = None


def get_set_recent_tags(set_tags:bool = False):
    '''
    Get the Recent tags or set after successful Creation of Questions
    '''
    try:
        username = st.session_state['Username']
        ERROR = f"No entry present for : {username}"
        worksheet = sh.worksheet("RecentValues") # Put it in memory?
        cell = worksheet.find(username, in_column = 1)
        if set_tags:
            values = [username]
            for tag_name in MAPPING.values():
                values.append(st.session_state[tag_name])
            
            if cell is not None: worksheet.delete_rows(cell.row)
            worksheet.append_row(values)
            return st.session_state

        if cell is not None:
            recent_values = worksheet.row_values(cell.row)[1:]
            for i in range(6 - len(recent_values)): recent_values.append('') # Because row_values() ignores trailing empty and there are 6 tags
            for index, tag_name in enumerate(MAPPING.values()):
                st.session_state[tag_name] =  recent_values[index]
            return st.session_state
        
        else: st.error(ERROR, icon="🚫")
    except Exception as e:
        st.error(ERROR, icon="🚫")
        print_exc()


def process_pdf(): # part of First Flow
    '''
    Givne a PDf, extract images from it
    '''
    pdf = st.session_state['uploaded']
    
    in_pdf = pdf.read()
    dir_path = pdf.name.split(".pdf")[0]

    if os.path.exists(dir_path): # remove directory if exists
        shutil.rmtree(dir_path)
    

    _ = extract_images_from_pdf(in_pdf, dir_path)


    st.session_state['file_dir'] = dir_path
    st.session_state['image_file_paths'] = sorted([os.path.join(st.session_state['file_dir'],i ) for i in os.listdir(st.session_state['file_dir']) if is_image(i)], 
                                            key = image_name_sorting)
    
    resize_all_images(st.session_state['image_file_paths']) # resize images to full HD
    image = Image.open(st.session_state['image_file_paths'][st.session_state['image_counter']])
    if image.mode == "RGBA": image = image.convert("RGB")
    st.session_state['display_image'] = np.array(image)

    st.session_state['zipped_path'] = shutil.make_archive(dir_path, 'zip', dir_path)


def process_zip(zip_file): # part of second flow
    '''
    Extract the zip file uploaded
    '''
    st.session_state['file_dir'] = zip_file.name.split(".zip")[0]

    if os.path.exists(st.session_state['file_dir']): # if the directory exists, clean the directory
        shutil.rmtree(st.session_state['file_dir'])

    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(st.session_state['file_dir'])

    st.session_state['image_file_paths'] = sorted([os.path.join(st.session_state['file_dir'],i ) for i in os.listdir(st.session_state['file_dir']) if is_image(i)], 
                                            key = image_name_sorting)
    
    
    resize_all_images(st.session_state['image_file_paths']) # resize images which are bigger than Full HD 1920
   
    image = Image.open(st.session_state['image_file_paths'][st.session_state['image_counter']])
    if image.mode == "RGBA": image = image.convert("RGB")
    st.session_state['display_image'] = np.array(image)
  

def create_sidebar():
    '''
    Render these parts when Zip File is uploaded. 
    1. Upload Zip File
    2. Unzip and process Zip file
    '''
    with st.sidebar:
        st.markdown("#### Enter Username & Password")

        col1, col2 = st.columns(2)
        with col1:
            _ = st.text_input("Username", key = "Username", help = "Enter username for which you want to run the pipeline")
        
        with col2:
            _ = st.text_input("Password", key = "Password", help = "Password for the above username",)


        st.markdown("#### Enter no of columns")
        st.number_input("No of columns", min_value= 1, max_value = 3, key="num_columns", label_visibility="collapsed", on_change = reset_vertical_lines_callback) # Numbers of columns in image

        st.markdown("_"*10)
        if st.button(label = "Click to use the most recently used tags", key = "recent_tags", disabled = False if st.session_state["Username"] else True):
            get_set_recent_tags()

        with st.expander("Click & expand to enter the values of tags you already know", expanded=False):
            for tag_name in MAPPING.keys():
                _ = st.text_input(tag_name, key = MAPPING[tag_name], placeholder = "")

        try:
            st.markdown("_"*10)
            st.write("Please fill the following details: ([Source]({}))".format(get_teacher_name_worksheet_url()))
            st.session_state["DIGITISING_FOR_TEACHER"] = st.selectbox("""Who are you digitising for?""", get_teacher_name_tuple())
            st.session_state["DIGITISED_BY"] = st.selectbox("""Who are you?""", get_digitiser_name_tuple())
            st.session_state["EXPORT_DATA_SHEET"] = st.selectbox("""What sheet would you like to see the metadata in?""", get_export_worksheet_names())
            update_export_sheet(st.session_state["EXPORT_DATA_SHEET"])
            # Somehow, updating the selectbox runs this piece of code again.
            st.session_state["WRITE_MAPPING_DATA"] = st.checkbox("Create Mapping Data?", value=True)
            if not st.session_state["WRITE_MAPPING_DATA"]:
                st.markdown("""Mapping Data will <span style="color:red"><b><i>not</i></b></span> be created""", unsafe_allow_html=True)

        except Exception as e:
            st.warning("Something went wrong while loading the legends list: " + str(e))
            print_with_date(e)
        
        st.markdown("_"*10)
        st.markdown("#### Upload Chapter Sheet")
        st.session_state["chapter_sheet"] = st.file_uploader("Chapter Sheet", type = ["csv", "xls", "xlsx"], label_visibility="hidden")


def second_flow_run_pipelines():
    '''
    Show Image and Run Pipeline button
    '''
    show_image_flow() # render image

    if st.session_state["col_detection"] is None:
        st.session_state["col_detection"] = get_cols_data(st.session_state['image_file_paths'])
    
    st.warning(f"Single / Multi Column stats:  {st.session_state['col_detection']}", icon="🚩")

    st.markdown(f"""<h5 style="color:#000000">Click on the button below to start generating questions</h5>""", unsafe_allow_html=True)
    if st.button("Run Pipeline"):
        process_second_part(st.session_state['file_dir']) # digitise questions from images


def process_second_part(file_dir):
    '''
    Process image and push questions to DB depending on whether they are MCQ or Short Long Answer
    '''
    try:
        TOKEN, _  = connect_DB(st.session_state['Username'], st.session_state['Password'])
    except Exception as e:
        st.error("Authentication Error: Enter correct Username and Password")
        st.stop()


    with st.spinner("Removing header, footer and splitting images in columns..."):
        crop_images()

    run_mixed_pipeline(st.session_state, TOKEN, ADD_QUESTION_URL, MATHPIX_API_ID, MATHPIX_API_KEY) 
    get_set_recent_tags(set_tags=True) # Set all the tags in the session state to a sheet

    st.balloons()
    st.success("""Questions added Successfully!! Please REFRESH the page now to digitise another one""", icon="✅")
    st.warning("""Please find the allotment sheet entries **[here]({})**""".format(get_allotment_worksheet_url()), icon="❗")

    # clear Directory and Zip file if present
    if os.path.exists(file_dir): # remove directory if exists
        shutil.rmtree(file_dir)

    if os.path.exists(file_dir+".zip"): # if there's a zip file that exists
        os.remove(file_dir+".zip")


def load_fresh_image():
    '''
    If current value of header is smaller than the previous value OR vertical lines have moved, render new image because all the pixels have been overridden 
    with colour and can't be undone
    '''
    result = False

    if (st.session_state['header'] < st.session_state['image_points'][0]) or\
        (st.session_state['footer'] < st.session_state['image_points'][1] or\
        (st.session_state['Line 1'] != st.session_state['image_points'][2]) or\
        (st.session_state['Line 2'] != st.session_state['image_points'][3])):
        
        result = True
    
    st.session_state['image_points'][0] = st.session_state['header'] 
    st.session_state['image_points'][1] = st.session_state['footer']
    st.session_state['image_points'][2] = st.session_state['Line 1']
    st.session_state['image_points'][3] = st.session_state['Line 2']

    return result


def display_head_foot_area(image:Union[np.ndarray, str], header, footer):
    '''
    Remove or display only the header and footer
    '''
    if footer == 0: footer = -1

    COLOR = (219, 199, 20)

    if isinstance(image, str):
        image = Image.open(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.array(Image.open(image))

    image[:header,:,:] = COLOR
    image[footer:,:,:] = COLOR
    
    return Image.fromarray(image)


def save_image_crops(image_path, header, footer, line_1 = None, line_2 = None):
    '''
    1. Remove header and footer
    2. If No of columns > 1, split the page in different columns vertically and save each crop
    3. Delete the original image
    '''
    split_names = []
    ext = "."+image_path.split(".")[-1]

    crops = []
    if isinstance(image_path, str):
        image = Image.open(image_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")

    w, h = image.size

    if footer <= 0: footer = h + footer # in case there's negative value, which is -30 by default, just change that value

    if line_1: # Split the image in number of columns and then ave the splits individually,  the original image in a new directory
        crops.append(image.crop((0,header,line_1,footer)))

        if not line_2: # split in 2 parts if there is just 1 line
            crops.append(image.crop((line_1,header,w, footer)))
        
        else: # split in 3 parts
            crops.append(image.crop((line_1,header,line_2, footer)))
            crops.append(image.crop((line_2,header,w,footer)))

        for i,crop in enumerate(crops):
            crop_path =  image_path.replace(ext, f"-<SPLIT-{i}>{ext}")
            crop.save(crop_path)
            split_names.append(crop_path) # For BB Mapping Purpose

        # don't delete the original image as we need to save it in DB in original form
        image_new_path = os.path.join(st.session_state['file_dir'],"original_images",image_path.split('/')[-1])
        os.rename(image_path, image_new_path) # push to other locations for future reference

        st.session_state['image_splits_mapping'][image_new_path] = split_names  # Used for Bounding Box patching up

    else: # 1 column means the original image == cropped image so it isn't sent to the 'original_image'
        image.crop((0,header,w,footer)).save(image_path)
        st.session_state['image_splits_mapping'][image_path] = [image_path]


def crop_images():
    '''
    Apply Header footer removal + Vertical splits, if >1 columns
    '''
    line_1 = None
    line_2 = None
    header, footer = st.session_state['header'], st.session_state['footer']

    if st.session_state['num_columns'] >= 2:
        line_1 = st.session_state["Line 1"]
        os.makedirs(os.path.join(st.session_state['file_dir'],"original_images"), exist_ok=True) # keep original images here instead of deleting them

    if st.session_state['num_columns'] == 3: line_2 = st.session_state["Line 2"]

    _ = [save_image_crops(i, header, footer, line_1, line_2) for i in st.session_state['image_file_paths']] # split images
    st.session_state['image_file_paths'] = sorted([os.path.join(st.session_state['file_dir'],i ) for i in os.listdir(st.session_state['file_dir']) if is_image(i)],
                                            key = image_name_sorting)


def show_column_lines(image, line_1 = None, line_2 = None):
    '''
    Put Vertical Column line on Image for display purpose
    '''
    if isinstance(image, str):
        image = Image.open(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")

    elif isinstance(image, Image.Image):
        image = np.array((image))

    if line_1:
        image[:,line_1:line_1+3,:] = (255,0,0)
    if line_2:
        image[:,line_2:line_2+3,:] = (0,0,255)
    
    return Image.fromarray(image)


def reset_vertical_lines_callback():
    '''
    Set Vertical Column split Lines values based on the dimensions of the image ccurrently being displayed
    '''
    def whole_num(num):
        x = num / 100
        deci, inte = modf(x)
        if deci >= 0.5: return int((inte+1) * 100)
        return int(inte * 100)

    h = st.session_state['display_image'].shape[1]

    if st.session_state['num_columns'] == 2:
        st.session_state["Line 1"] = h//2
        st.session_state["Line 2"] = None

    elif st.session_state['num_columns'] == 3:
        portion = h//3
        st.session_state["Line 1"] = whole_num(portion)
        st.session_state["Line 2"] = whole_num(portion * 2)
    
    else:
        st.session_state["Line 1"] = None
        st.session_state["Line 2"] = None

    
def process_image():
    '''
    Process 1 image for Display only which shows vertical lines + header footer coloured pixels
    '''
    if load_fresh_image(): # load new image only when we have values changes
        image = Image.open(st.session_state['image_file_paths'][st.session_state['image_counter']])
        if image.mode == "RGBA": image = image.convert("RGB")
        st.session_state['display_image'] = np.array(image)
    
    image = display_head_foot_area(st.session_state['display_image'], st.session_state['header'], st.session_state['footer'])
    image = show_column_lines(image, st.session_state["Line 1"], st.session_state["Line 2"])
    return image


def show_image_flow():
    '''
    Flow to Render Image, setting margins etc
    '''
    st.markdown("""Enter <span style="color:red"><b><i>Absolute Coordinates</i></b></span> (Horizontal Coordinate in below image) to Remove Header and Footer""", unsafe_allow_html=True)
    head_col, foot_col = st.columns(2)
    
    with head_col:
        st.number_input("Header (where it ENDS)", min_value = 0, step = 10, value = 0, key = "header")
    
    with foot_col:
        st.number_input("Footer (where it STARTS)", value = 0, min_value = 0, step = 10, key = "footer")


    st.markdown("_"*30)
    
    _ , prev_image, entry, next_image, _ = st.columns([3,4,2,4,3])

    with prev_image:
        st.write("Showing image number")
    
    with entry:
        image_num = st.number_input("Image Number", min_value = 1, step = 1, max_value=len(st.session_state['image_file_paths']), label_visibility= "collapsed")
        if image_num:
            st.session_state['image_counter'] = image_num - 1
            image = Image.open(st.session_state['image_file_paths'][st.session_state['image_counter']])
            if image.mode == "RGBA": image = image.convert("RGB")
            st.session_state['display_image'] = np.array(image)
            reset_vertical_lines_callback()
    
    with next_image:
        st.write(f"out of {len(st.session_state['image_file_paths'])} images")
        

    if st.session_state["num_columns"] > 1:
        st.markdown("""Set <span style="color:black"><b><i>Column Lines</i></b></span> (Vertical Coordinate)""", unsafe_allow_html=True)
        
        col_lines, image_col = st.columns([1,7])

        with col_lines:

            C = [None, "red", "blue"]
            for i in range(1,st.session_state["num_columns"]):
                label = f"Line {str(i)}"
                st.markdown(f"""<span style="color:{C[i]}"><b><i>{label}</i></b></span>""", unsafe_allow_html=True)

                st.session_state[label] = st.number_input(label, value = st.session_state[label], min_value = 0, step = 10, label_visibility="collapsed")
    
    
    else: image_col = st.container()
    
    with image_col:
        image = np.array(process_image())
        fig = px.imshow(image, width = 720, height = 1280, labels=dict(x="Vertical (Columns)", y="Horizontal (Header & Footer)",))
        st.plotly_chart(fig,use_container_width=True)