"""
Pipeline to digitise both SA-LA and MCQ questions automaticall. Uses the single page OCR flow
"""
from datetime import datetime

import pytz
from google_sheet_helpers import append_image_ids_to_google_sheet_with_chapters, flatten_labels_multiple_chapters
from jackett_webservice_helpers import create_mapping_data_jackett_webservice
from sa_la_helpers import  (QuestionOptionRegex, ANSWER_REGEX, is_mcq, split_ques_options, post_single_image_db, get_layout_type, get_tag_value,
                            QuestionGeneration, CreateBB)

from helpers import (get_mathPix_OCR, get_suggested_tagsV2, callAddQuestionAPI, insert_image_bb_data, image_name_sorting, 
                    handle_diagrams, process_chapter_topic_sheet, show_system_stat, resize, cleanText, preprocess_text, print_with_date,
                    st, re, np, os, pd, BeautifulSoup)

from traceback import print_exc
from stqdm import stqdm
import gc

def run_mixed_pipeline(session_state:dict, TOKEN, ADD_QUESTION_URL:str, MATHPIX_API_ID:str, MATHPIX_API_KEY:str):
    '''
    1. Run MathPix to get OCR on all images
    2. Generate Questions and Answers
    3. Identifiy if Question is MCQ or SA-LA
    4. Post each ORIGINAL image to our backend to get imageKey
    5. Get Suggested Labelling tags for each Question
    6. Push each question to DB
    '''

    add_question_header = { "accept": "application/json",
            "content-type": "application/json",
            'authorization': "Bearer " + str(TOKEN)
          }

    Q_GEN = QuestionGeneration()
    obj_bb = CreateBB(cols = st.session_state['num_columns'], line_1 = st.session_state["Line 1"], line_2 = st.session_state["Line 2"])
    
    
    with st.spinner("Trying to find if there's any pattern in the data for Ques- Ans"):
        total_images = len(st.session_state['image_file_paths'])
        regex_test_data = []

        for i in np.random.choice(st.session_state['image_file_paths'], min(7,total_images), replace = False):
            try:
                ocr = get_mathPix_OCR(i, MATHPIX_API_ID, MATHPIX_API_KEY)['html']
                regex_test_data.append(ocr)
            except Exception as e:
                print_with_date(f"MathPix data return error for {i} during regex testing : {e}")
                print_exc()
                continue

        QUESTION_REGEX, statement, selected_regex = Q_GEN.select_priority_regex(regex_test_data)
        if statement:
            st.info(statement,icon="ℹ️")
            del i, regex_test_data
        else:
            st.error("Unable to find any Ques - Ans pattern in the data, might be a new pattern. Send PDF along with the screenshot to DS Team and use MCQ or CD Pipeline for now",icon="❌")
            st.stop()

    error_pages = {"page":[],"cause":[]} # where no OCR data / pattern was found
    for page_num, image_path in stqdm(enumerate(sorted(st.session_state['image_file_paths'], key = image_name_sorting)), total = total_images, desc = "Generating Questions + BB"): # image splits to send to MathPix
        if obj_bb.cols == 1: # base_dir/image_name.EXTENSION
            split_num = 0
            original_image_name = image_path

        else: # basedir/image_name-<SPLIT-x>.EXTENSION
            first_part, second_part = image_path.split("-<SPLIT-")
            ext = second_part[2:] # Image extenstion as  it'll be 0>.png 1>.jpeg etc 
            split_num = int(second_part[0]) # Gives you a number 0>.png gives 0, 1>.jpeg gives 1 etc
            base, image_name = first_part.split("/")
            original_image_name = os.path.join(base,"original_images",image_name+ext)


        ocr_returned = True
        try:
            mathpix_response = get_mathPix_OCR(image_path, MATHPIX_API_ID, MATHPIX_API_KEY)
        except Exception as e:
            error_pages["page"].append(original_image_name)
            error_pages["cause"].append("MathPix didn't return Any Data")
            print_exc()
            continue
        
        if "error" in mathpix_response:
            st.warning(f"Error | {mathpix_response['error']} | returned by MathPix instead of OCR for image: {image_path}")
            error_pages["page"].append(original_image_name)
            error_pages["cause"].append(mathpix_response['error'])
            continue
        
        if "html" in mathpix_response:
            page_text = mathpix_response['html']
        else:
            ocr_returned = False
            for res in [1280, 720, 480, 224]:
                resize(image_path, res).save(image_path)
                mathpix_response = get_mathPix_OCR(image_path, MATHPIX_API_ID, MATHPIX_API_KEY)
                if "html" in mathpix_response:
                    page_text = mathpix_response['html']
                    ocr_returned = True
                    break

        if ocr_returned:   
            if "line_data" in mathpix_response: line_data = mathpix_response['line_data']
            else: line_data = None
        
        else:
            st.warning(f"MathPix Error: No OCR Data returned by Mathpix for: {image_path}. It'll be skipped and you have to use CD Pipeline", icon="⚠️")
            error_pages["page"].append(original_image_name)
            error_pages["cause"].append("No Text found in page")
            continue
        
        page_text = preprocess_text(page_text, selected_regex) # preprocess text for nested <li> and numbered list to numbered question
        patterns = QUESTION_REGEX.findall(page_text)
        
        # if bigger priority regex fails, check if there is another regex that satisfies here
        if (not patterns) and (selected_regex in ["q3", "q4"]): # in case it fails for once only. Happens with q3 so if it fails, try to see if q4  is satisfied here
            print_with_date(f"First Test failed, running Second Test for image {image_path}")
            page_text = preprocess_text(page_text) # Preprocess basic text
            temp_q_regex, _, _ = Q_GEN.select_priority_regex([page_text]) # test once again
            
            if temp_q_regex: # if second test succeds
                patterns = temp_q_regex.findall(page_text)
                Q_GEN.generate_qa(page_num, page_text, image_path, original_image_name, split_num, temp_q_regex, ANSWER_REGEX)
            
            else: # if second test fails
                patterns = [] # in case No pattern Found, means either question - Answers are spanning or there's just garbage test -> We can delete the question
                Q_GEN.generate_qa(page_num, page_text, image_path, original_image_name, split_num, QUESTION_REGEX, ANSWER_REGEX) # using defult regex -> 0 patterns found
                print_with_date(f"Failed Second Test too. No pattern found in {image_path}")

        else: # if the second pattern fails or no pattern at all, just pass the whole page text to the generator. It'll either process this as a remaining Question or Answer
            Q_GEN.generate_qa(page_num, page_text, image_path, original_image_name, split_num, QUESTION_REGEX, ANSWER_REGEX)
            
        obj_bb.generate_bboxes(line_data, patterns, original_image_name, split_num)
    

    del original_image_name, patterns
    if Q_GEN.ques[0] == [None, None, None]:
        Q_GEN.ques.pop(0) # in case it is Emptty, pop it. It is set to empty just to avoid the error when there is no Regex detection on the first page
        Q_GEN.ans.pop(0) # because it'll hit assertion error

    with st.spinner("Mapping Bounding Boxes to Questions"):
        Q_GEN.map_ques_to_bb_diag(obj_bb) # map Questions to Bounding Box and BBoxes of Diagrams

    q, a = len(Q_GEN.ques), len(Q_GEN.ans)
    empty_q = sum([i[0]=="" for i in Q_GEN.ques])
    empty_ans = sum([i=="" for i in Q_GEN.ans])

    if q != a: 
        st.error(f"Different numbers of questions {q} and answers {a} found. This is not expected. Send PDF to DS team and use MCQ or CD pipeline if urgent.", icon="❌")
        st.stop()
    
    st.info(f"{str(q)} Questions - Answers pair found out of which {str(empty_q)} Questions and {str(empty_ans)} Answers are empty",icon="ℹ️")


    image_push_header = {'authorization': 'Bearer '+TOKEN, "accept": "application/json", "content-type": "application/json"}
    imageKey_mapping = {}

    num_mcq = 0
    push_bbox_image = None # stores the image name. Once a new Image starts, pushes the old one's BBoxes to DB. Reduces error rate

    chapter_sheet_mapping = None
    topic_sheet_mapping = None
    chapter_page_pointer = None
    topic_page_pointer = None
    chapter_original_image_num_hash = {} # store the {image_name: chapter} which has been processed already for saving computation cost
    topic_original_image_num_hash = {}

    if st.session_state["chapter_sheet"] is not None:
        chapter_sheet_mapping, chapter_page_pointer = process_chapter_topic_sheet(create = "chapter") # dictonary which has page number and it's chapter mapping
        topic_sheet_mapping, topic_page_pointer = process_chapter_topic_sheet(create = "topic")

    TEMPtags = {}  # check for labels from user + suggested tags
    chapter_image_id = {} # keep the list of chapter to source image ID mappings
    
    # Push Questions to DB one by one
    for i in stqdm(range(q), desc="Pushing Question to DB", total=q): # [(question, image_name), answer]
        if i % 75 == 0: # in the starting as well as every X no of questions
            gc.collect()
            show_system_stat(ram_only = True) # memory and Hard disk status

        options = [] # Empty -> Change in case it is MCQ
        question, original_image_path, split_num, _ , boundingBoxId, diag_bboxes = Q_GEN.ques[i]
        answer = Q_GEN.ans[i]
        
        try:
            if original_image_path not in imageKey_mapping:
                imageKey_mapping[original_image_path] = post_single_image_db(image_push_header, original_image_path) # push images to backend to get imageKey 

            sourceImageId = imageKey_mapping[original_image_path] # each image that was pushed to DB has a unique key

        
        except Exception as e:
            sourceImageId = None # otherwise it'll either use the last Id or give error if there was error on the first page
            ERROR = f"Error in Pushing Image to DB: {e} || Image {original_image_path} won't be present in webapp along with questions. {i} questions have been pushed already"
            st.warning(ERROR, icon="⚠️")
            print_with_date(ERROR)
            print_exc()
            pass
        
        que_options = is_mcq(QuestionOptionRegex, question)

        if que_options: # if there is a matched pattern of  Q-Option just see whether it is found in Answer and Options too
            ans_option = is_mcq(QuestionOptionRegex, answer) # whether the answer has MCQ pattern too

            # MCQ If -> if Answer is empty || Answer has no options structure || No oof options found in Question != No of Options found in answer
            if (not answer) or (not ans_option) or (len(ans_option) != len(que_options)):
                question, options = split_ques_options(que_options)
                q_type = "MCQ"
                num_mcq += 1
            
            else:
                q_type = "OPEN_SHORT"
        else:
            q_type = "OPEN_SHORT"

        # handle diagrams and images for question and options
        num_options = len(options)
        q_diag_ids, opt_diag_ids = handle_diagrams(diag_bboxes, original_image_path, q_type, image_push_header, num_options=num_options, 
                                                    header_margin=st.session_state['header'])
        
        for index, im_id in enumerate(opt_diag_ids):
            options[index]['images'].append({'_id': index, 'imageKey': im_id})
        
        question_images = []
        for index, im_id in enumerate(q_diag_ids):
            question_images.append({'_id': index, 'imageKey': im_id})
        
         
        ans = {
        "answerId": 0,
        "answerText": cleanText(answer),
        "images": []
        }

        questionTemplateJackettWebservice = {
            "marks": 0,
            "questionDifficulty": "MEDIUM",
            "autoMarking": False,
            "showWorkingOut": False,
            "images": question_images,
            "isChild": False,
            "parentQuestionid": 0,
            "childQuestions": [],
            "keywords": [],
            "boundingBoxId": "",
            "suggestedLabellingId": "",
            "sourceImageId": "",
            "questionType": q_type
            }
        
        try:
            suggested_labels, suggestedLabellingId = get_suggested_tagsV2(question)

            TEMPtags['chapter'], chapter_page_pointer = get_tag_value(chapter_sheet_mapping, chapter_page_pointer, chapter_original_image_num_hash, 
                                                                        original_image_path, 'chapter',suggested_labels["chapter"])
            
            TEMPtags['topic'], topic_page_pointer = get_tag_value(topic_sheet_mapping, topic_page_pointer, topic_original_image_num_hash, 
                                                                        original_image_path, 'topic','')
            
            
            # Adding chapter tags for allocation automation
            if TEMPtags['chapter'] not in chapter_image_id:
                chapter_image_id[TEMPtags['chapter']] = {}
            if sourceImageId not in chapter_image_id[TEMPtags['chapter']]:
                chapter_image_id[TEMPtags['chapter']][sourceImageId] = 1
            else:
                chapter_image_id[TEMPtags['chapter']][sourceImageId] += 1

            if session_state['subjectTags'] != '':
                TEMPtags['subject'] = session_state['subjectTags']
            else:
                TEMPtags['subject'] = suggested_labels["subject"]
            
            if session_state['curriculum'] != '':
                TEMPtags['curriculum'] = session_state['curriculum']
            else:
                TEMPtags['curriculum'] = suggested_labels["curriculum"]
            
            if session_state['classes'] != '':
                TEMPtags['classes'] = [session_state['classes']]
            else:
                TEMPtags['classes'] = [suggested_labels["class"]]

            # labels not suggested by ds-backend -> Either entered or Empty
            if session_state['sources'] != '':
                TEMPtags['source'] = [session_state['sources']]
            else:
                TEMPtags['source'] = ''


        except Exception as e:
            st.warning(f"Suggested Labelling API Error: {e} || Tags for question {i} will be set to <EMPTY STRING>. Send Screenshot to Backend, DS team", icon="⚠️")
            print_exc()

            if session_state['chapter'] != '':
                TEMPtags['chapter'] = session_state['chapter']
            else:
                TEMPtags['chapter'] = ''

            if session_state['subjectTags'] != '':
                TEMPtags['subject'] = session_state['subjectTags']
            else:
                TEMPtags['subject'] = ''
            
            if session_state['curriculum'] != '':
                TEMPtags['curriculum'] = session_state['curriculum']
            else:
                TEMPtags['curriculum'] = ''
            
            if session_state['classes'] != '':
                TEMPtags['classes'] = [session_state['classes']]
            else:
                TEMPtags['classes'] = ''

            # labels not suggested by ds-backend -> Either entered or Empty
            if session_state['sources'] != '':
                TEMPtags['source'] = [session_state['sources']]
            else:
                TEMPtags['source'] = ''

            if session_state['topic'] != '':
                TEMPtags['topic'] = session_state['topic']
            else:
                TEMPtags['topic'] = '' 

        
        # First Pushing Bounding Box data when there is a difference in the current and the variable's name
        if (push_bbox_image is not None) and (push_bbox_image != original_image_path):
            try:
                layout_type = get_layout_type(obj_bb.cols, q_type)
                if push_bbox_image not in imageKey_mapping:
                    imageKey_mapping[push_bbox_image] = ''
                    print_with_date(f"Unable to push {push_bbox_image} to DB. Setting to EMPTY")

                insert_image_bb_data(imageKey_mapping[push_bbox_image], push_bbox_image.split("/")[-1], obj_bb.image_bb_mapping[push_bbox_image],
                                                                                                        st.session_state['header'], layout_type)
            except Exception as e:
                ERROR = f"Error : {e} || Bounding Boxes for Image {push_bbox_image} having 'sourceIageId': {imageKey_mapping[push_bbox_image]} won't be present for the image"
                st.warning(ERROR, icon="⚠️")
                print_with_date(ERROR)
                print_exc()
                pass
            
        push_bbox_image = original_image_path

        try:
            # Push the new Question to DB
            addQuestionResp = callAddQuestionAPI(questionTemplateJackettWebservice, ADD_QUESTION_URL, add_question_header, 
                username = session_state['Username'],
                author = session_state['Username'],
                questionText = cleanText(question),
                options = options,
                answers = [ans],
                tags = TEMPtags,
                boundingBoxId = boundingBoxId,
                suggestedLabellingId = suggestedLabellingId,
                sourceImageId = sourceImageId
            )
        except Exception as e:
            ERROR = f"Error : {e} || addQuestionAPI failed. Question {i} from Image: {original_image_path} won't be present in the DB"
            st.warning(ERROR, icon="⚠️")
            print_with_date(ERROR)
            print_exc()
            pass


        Q_GEN.ques[i] = None # save memory by setting it to None
        Q_GEN.ans[i] = None

    # Push last processed image's BBoxes
    try:
        layout_type = get_layout_type(obj_bb.cols, q_type)
        if push_bbox_image not in imageKey_mapping:
            imageKey_mapping[push_bbox_image] = ''
            print_with_date(f"Unable to push {push_bbox_image} to DB. Setting to EMPTY")
            
        insert_image_bb_data(imageKey_mapping[push_bbox_image], push_bbox_image.split("/")[-1], obj_bb.image_bb_mapping[push_bbox_image],
                                                                                                st.session_state['header'], layout_type)
    except Exception as e:
        ERROR = f"Error : {e} || Bounding Boxes for Image {push_bbox_image} having 'sourceIageId': {imageKey_mapping[push_bbox_image]} won't be present for the image"
        st.warning(ERROR, icon="⚠️")
        print_with_date(ERROR)
        print_exc()
        pass


    st.info(f"Out of {str(q)} Questions, {str(num_mcq)} were found as MCQ and {str(q-empty_q-num_mcq)} were SA-LA",icon="ℹ️")

    download_df = pd.concat([pd.Series(imageKey_mapping.values(), name = "Present Image Ids"), pd.DataFrame(error_pages)], axis = 1).to_csv(index = None).encode("utf-8")
    st.download_button(label="Download CSV",data=download_df, file_name=st.session_state["file_dir"]+".csv", mime='text/csv',)

    # uploading the data to google sheet
    try:
        current_time = datetime.now(pytz.timezone("Asia/Kolkata"))
        created_at = current_time.strftime("%d-%m-%Y %H:%M:%S ") + current_time.tzname()
        addedMetadata = {"Digitised By": st.session_state["DIGITISED_BY"], "Created At": created_at}
        flattenedLabels = flatten_labels_multiple_chapters(TEMPtags, chapter_image_id, session_state['Username'], session_state["DIGITISING_FOR_TEACHER"])
        append_image_ids_to_google_sheet_with_chapters(flattenedLabels, **addedMetadata)
    except Exception as e:
        st.info(f"Allotment sheet could not be updated. Please do it Manually", icon="⚠️")
        print_exc()
    # Adding mapping data on the backend
    try:
        if st.session_state["WRITE_MAPPING_DATA"]:
            create_mapping_data_jackett_webservice(flattenedLabels, q_type)
    except Exception as e:
        st.info(f"Mapping data could not be added in the DB. Please do it Manually", icon="⚠️")
        print_exc()