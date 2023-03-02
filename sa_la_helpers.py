"""
All the helpers to process and handle data related to Sa-La
"""
from bs4 import BeautifulSoup
import re
from helpers import *

ANSWER_REGEX = re.compile(r'((?:\n|<div>|<br>|<br/>)(?:हल|उत्तर|தீர்வு|Answer|Solution|Ans.|Sol.)(?:\s*:{0,1}\s*\.*\d*\s*))')
QuestionOptionRegex = r"((?:.|\n|\r)*)\<br\>\n?\r?\(?[1|A|a]\)?((?:.|\n|\r)*)\<br\>\n?\r?\(?[2|B|b]\)?((?:.|\n|\r)*)\<br\>\n?\r?\(?[3|C|c]\)?((?:.|\n|\r)*)\<br\>\n?\r?\(?[4|D|d]\)?((?:.|\n|\r)*)"


def is_mcq(QuestionOptionRegex:str, question_text:str) -> list:
    '''
    Find if a question is MCQ. If it is, return list of [Question, [Options]] else return None
    '''
    if not question_text: return None
    matches = re.findall(QuestionOptionRegex, question_text, re.MULTILINE)
    if matches: return matches[0]
    else: return None



def mark_answer(answer:str):
    '''
    Mark the Answer
    '''
    ans = 10
    if '(a)' in answer: ans = 0
    elif '(b)' in answer: ans = 1
    elif '(c)' in answer: ans = 2
    elif '(d)' in answer: ans = 3
    
    return ans


def split_ques_options(questionOptionMatches:list, answer:str = None):
    '''
    Given a List of found Question-Options, try to filter Question and Options seperately
    '''
    if not questionOptionMatches: raise NotImplementedError("Input can't be None or Empty List. Check the code flow")

    if len(questionOptionMatches) != 0 and len(questionOptionMatches)>=4:
        question_text = questionOptionMatches[0]
        options = [questionOptionMatches[1], questionOptionMatches[2], questionOptionMatches[3], questionOptionMatches[4]]

        TEMPoptions = []
        ctrOption = 0
        answerIdMapping = 10 if not answer else mark_answer(answer[:10]) # assuming first 10 chars have the (a) , (b) etc

        for option in options:  # options
            temp_single_option = {
                "optionId": ctrOption,
                "optionText": cleanText(option),
                "isAnswer": True if ctrOption == answerIdMapping else False,
                "isAttempted": False,
                "images":[]
            }
          
            ctrOption += 1
            TEMPoptions.append(temp_single_option)
        
        return question_text, TEMPoptions


def get_tag_value(sheet_mapping:dict, page_pointer:int, original_image_num_hash:dict, original_image_name:str, required_tag:str, suggested_labelling_value:str):
    '''
    3 ways to set the chapter / Topic tags and priority is Given as: Excel Sheet > Input Box > Suggested Labelling
    args:
        sheet_mapping: Stores {page_num: chapter_name}. page_num here is the end limit after which a new chapter starts
        page_pointer: Points to the current page limit. 15 means that any page number <= this refers to the key 15 in sheet_mapping
        original_image_num_hash: Dictonary which stores the page numbers which has been passed already. Save computation in 2,3 cols + multi Que per image
        original_image_name: Name of the original whose numbering will decide the chapter
        required_tag: Tag name whose value you have to get from session state
        suggested_labelling_value: Chapter  / Topic Suggested from suggested labelling
    '''
    if sheet_mapping is not None: # first priority
        if original_image_name in original_image_num_hash: return original_image_num_hash[original_image_name], page_pointer
        
        num = int(re.findall(r"\d+",original_image_name)[-1]) # assumes image name's unique idetifier is an incremental number in the end
        if num > page_pointer: # if there is a new page which exceeded the limit, set a new page_pointer
            for key in sheet_mapping.keys():
                if key > page_pointer:
                    page_pointer = key
                    break
        
        original_image_num_hash[original_image_name] = sheet_mapping[page_pointer]
        return sheet_mapping[page_pointer], page_pointer

    elif st.session_state[required_tag] != '': return st.session_state[required_tag], page_pointer # second priority
    else: return suggested_labelling_value, page_pointer


class QuestionGeneration:
    def __init__(self):
        self.ques = [[None, None, None]] # [Question_text, image_name, split_num, bbox_coordinates, bbox_id, [diag_coors1, diagg_coors_2...]]
        self.ans = [""]
    
    def select_priority_regex(self, page_texts:list):
        '''
        Test Regexes based on the criteria to control False Positives
        Note: It can fail in case there wasn't any text resembling on the specific page OR the highest priority has 0 matches in the specific page but is present all over the book
        
        If the test fails with one page, you can pass say, a total of N pages randomly and see if it captures something
        '''
        q1 = re.compile(r'((?:\n|<div>|<br>|<br/>)(?:प्रश्न|उदाहरण|கேள்வி|Question|Example|Problem)(?:\s*:{0,1}\s*\d+\.{0,1}\s*))') # CAN be a space at the end. Less strict as finding something with Whole word with a number is rare
        q2 = re.compile(r'((?:\n|<div>|<br>|<br/>)(?:Ques|Que|Prob|Ex|Q)(?:\s*\.{0,1}\s*\d+\.{0,1})(?:\s+|<br>|</br>|<br/>))') # a bit strict as there
        q3 = re.compile(r'((?:\n\d+\.)(?:\s+|<br>|</br>|<br/>))') # Very specific while dealing with these as there can be lots of numbers. \n123.<SPACE> means the question will be in the beginning only
        q4 = re.compile(r'(\n<li>)') # Negative with List elements if present
    
        mapping = {"q1":q1, "q2":q2, "q3":q3, "q4":q4}
        print_mapping = {"q1":"<NEWLINE_STARTS>उदाहरण|Question|Example|Problem <NUM>",
                        "q2":"<NEWLINE_STARTS>प्रश्न|Ques|Que|Prob|Ex|Q <NUM><SPACE>", 
                        "q3":"<NEWLINE_STARTS><NUMBER><DOT><SPACE>", 
                        "q4":"<NEWLINE_STARTS><NUMBER><DOT><SPACE> but in form of <NEWLINE_STARTS><li>"}
        result = {}

        for index, page_text in enumerate(page_texts):
            for key in mapping.keys():
                if key == "q4" and "</ol>" not in page_text: continue # Consider <li> only if it is inside <ol>. As q4 is the last, you can use break
                if mapping[key].findall(page_text):
                    if key in result : result[key] += 1
                    else: result[key] = 1
                    break
        
        try:
            statement = f"Most Found Question Pattern in random pages: {print_mapping[max(result, key = result.get)]}"
            max_regex_key = max(result, key = result.get)
            return mapping[max_regex_key], statement, max_regex_key # return the compiled version of the regex
        except ValueError:
            print_with_date(f"Test Failed. no pattern Found in {index+1} random pages. Returning None")
            return None, None, None
    

    def re_split(self,regex, text):
        '''
        A way around to split the string because groups are also returned when used with re.split() and the original digitisation logic 
        had been written not to consider those
        '''
        return re.sub(regex, "<--SPLIT-->", text).split("<--SPLIT-->")


    def process_zero_index(self, first_part, answer_regex, image_name, split_num:int):
        '''
        Process the first part of any page
        First split of page can be one of the following: 
            1. Remaining previous ans :: handled by first if condition (len == 1)
            2. Starting of the previous ans :: handled by {elif -> if} condition
            3. Remaining previous que + ans :: handled in {elif -> else} condition below
            4. Starting of a fresh que (ideal scenario) :: it is handled in the first line and also in the parent loop as this type of text doesn't come
        '''
        if not first_part: raise Exception("How can first part be empty?")

        split = self.re_split(answer_regex, first_part) # split the text based if there is any ANSWER pattern is present
            
        if len(split) == 1: # Previous Answer is remaining because it has 1 split
            self.ans[-1] += ("\n"+split[0])
        
        elif len(split) >= 2: # Either it's a perfect Question OR (it's the remaining part OR extra text header)

            if split[0] == '': # it's a perfect question with no garbage text because it has 2 splits of q_ans_pair
                self.ans[-1] += ("\n" +split[1]) # add this text to previous answer

            else: # there was some text remaining 
                if self.ques[-1][0] is not None:
                    self.ques[-1][0] += ("\n" +split[0]) # add first part to previous question  :: Quetion is a tple of (question, image_name)
                else:
                    self.ques[-1] = [None,image_name, split_num]
                 
                self.ans[-1] += ("\n" +split[1]) # add second part to  previous answer which was set as '' ) 
                

            if len(split) > 2:
                print_with_date('MathPix Error || Multiple Answers found')

            for corrupted_ans in split[2:]: # in this scenario, the question tag was missing and there are 1+ answers present (there can't be 1+ question here as it'll be a new qa_pair split already)
                self.ques.append([None,image_name, split_num])
                self.ans.append(corrupted_ans) # we've dealt with the first 2 parts already which were of our interest. Now we're adding empty questions and their answers
                    

    def process_remaining_indices(self, qa_pair, answer_regex:str, image_name:str, split_num:int)-> tuple:
        '''
        Process the remaining Indices
        This can have:
            1. Full Question +  (H-F) Answer(s) Pair . (Half or full answer or answer can be there depending on what is the text of next page) AND (multiple answers can be there due to OCR error)
            2. Only Question (which can be full or half which we don't know until we look at the next page)
            3. Full Question + Half (or full) answer
        '''
        split = self.re_split(answer_regex, qa_pair) # create split of que and answers

        if len(split) == 0:
            raise Exception("How can split be of 0 length?")

        elif len(split) == 1: # No answer present
            self.ques.append([split[0], image_name, split_num])
            self.ans.append('') # there has to be an answer associated with it for sure so creating empty string of answer so that it can be handled in future loops

        elif len(split) >= 2: # length == 2 make a perfect question pair and >2 make a a pair with extra answers present
            self.ques.append([split[0], image_name, split_num])
            self.ans.append(split[1])

            if len(split) > 2:
                print_with_date('MathPix Error || Multiple Answers found')

                for corrupted_ans in split[2:]: # in this scenario, the question tag was missing and there are 1+ answers present (there can't be 1+ question here as it'll be a new qa_pair split already)
                    self.ques.append([None, image_name, split_num])
                    self.ans.append(corrupted_ans) # we've dealt with the first 2 parts already which were of our interest. Now we're adding empty questions and their answers
                    

    def generate_qa(self, page_num:str, page_text:str, image_path:str, original_image_name:str, split_num:int, question_regex, answer_regex)-> tuple:
        '''
        Generate Question Answers data. Image name is related to the page where question was found
        args:
            page_num: Index of the Page Number from a list of all the images (or image splts) which we have to send MathPix to get OCR
            page_text: 'html' response of the Page (or Page Split)
            image_path: Path of the Image (or image split) which has to be sent to the MathPix
            original_image_name: Name of the original image. If Columns == 1, it is same as the current image but in Col == [2,3], it is the original image 
            split_num: If Col == 1 , it is always 0 else what is the number of Split from the original image that we're processing
        '''
        if (page_text == '') or (page_text is None):
            return None

        QA_pairs = self.re_split(question_regex, page_text) # multiple splits of question and answers
        for index, q_ans_pair in enumerate(QA_pairs):
            
            if q_ans_pair == '': continue # there has been a perfect page start with text starting from Question so the first parrt is empty string

            if index == 0: # first split of every page is handled differently
                if page_num == 0: # first part of page number 1 of book is always the extra text, i.e chapter name, heading etc etc
                    continue
                
                self.process_zero_index(q_ans_pair, answer_regex, original_image_name, split_num)
                    
            else: # from second split onward, if present
                self.process_remaining_indices(q_ans_pair, answer_regex, original_image_name, split_num)


    def _map_ques_to_bbox(self, obj_bb) -> dict:
        '''
        Map each question in the entire PDF to it's own BBox
        args:
            obj_bb: Object of class CreateBB which ahs all the information about bounding boxes
        out:
            Dictonary which has the mapping for each split as {image_name:{split_num: [q_index1,q_index2]..}}
        '''
        prev_split = None
        prev_img_name = None
        q_bb_index = None
        temp_image_q_index = {} # map {'image_name':{split_num: [q_index1,q_index2]..}}

        for question_index, (q_text, image_name, split) in enumerate(self.ques): # traverse all the questions
            
            if (prev_split != split) or (prev_img_name != image_name): q_bb_index = 0

            # save the split number and question index according to the image name. It'll be used in mapping diagrams
            if image_name not in temp_image_q_index: temp_image_q_index[image_name] = {split:[question_index]}
            else:
                if split not in temp_image_q_index[image_name]: temp_image_q_index[image_name][split] = [question_index]
                else: temp_image_q_index[image_name][split].append(question_index)
            
            if q_text is None: # in case question was not found (due to OCR error but answer was there), it means no patterns was found means no BB will be there
                self.ques[question_index].extend([[], None, []]) # add [bbox_coors, bbox_id, diag_coors]
                
                prev_split = split # if an error comes in mapping, check if this is creating it
                prev_img_name = image_name
                continue
            
            try:
                q_bbox_coors, bbox_id = obj_bb.image_bb_mapping[image_name][split][q_bb_index] # (Bbox Coordinates , BBox ID)
            except IndexError:
                print_with_date(f"Image-BBox mapping error for image {image_name} with split {split}. Setting BBoxes to []")
                q_bbox_coors, bbox_id = [], None
         
            self.ques[question_index].append(q_bbox_coors) # add bbox data to the question
            self.ques[question_index].append(bbox_id)
            self.ques[question_index].append([]) # for diagrams appending

            prev_split = split
            prev_img_name = image_name
            q_bb_index += 1
        
        return temp_image_q_index


    def _map_ques_to_diag(self, image_q_index:dict, obj_bb):
        '''
        For each diagram in each image split, assign the diagram to the question which has highest area in common.
        args:
            image_q_index: Dictonary which has the mapping for each split as {image_name:{split_num: [q_index1,q_index2]..}}
            obj_bb: Object of class CreateBB which ahs all the information about bounding boxes
        '''
        for image_name in obj_bb.table_diagrams.keys(): # travesrse each image name -> runs for Number of images times
            for split in range(obj_bb.cols): # Max 3 Times
                diag_common_area = {} # save the best Intersection value, question_id for the each diagram

                # Check Each BBox to each Question BBOx in a specific split only and map to the question which has the most area in common
                for diag_index, diag_box in enumerate(obj_bb.table_diagrams[image_name][split]): # 1 diagram is supposed to be related to 1question only with the highest overlapping one
                    for question_index in image_q_index[image_name][split]: # image to questionid mapping (to consider questions only for a specific page)
                    
                        if self.ques[question_index][0] == '': continue # Empty Question -> no BB Box. If not handled, will give error
                        
                        q_bbox_coors = self.ques[question_index][3] # question bbox
                        if not q_bbox_coors: continue # empty box

                        intersection_area = obj_bb.intersection_area(diag_box,q_bbox_coors.copy(), split) # find common area
                    
                        if intersection_area > 0:
                            if diag_index not in diag_common_area:
                                diag_common_area[diag_index] = [intersection_area, question_index, diag_box]
                            else:
                                if intersection_area > diag_common_area[diag_index][0]: # if it is greater than the older entry, then the diag belongs to that question
                                    diag_common_area[diag_index] = [intersection_area, question_index, diag_box]

                
                    if diag_index in diag_common_area: # means if the diagram was not part of some question due to OCR error, it'll be skipped in this step else it would have given KeyError
                        _, q_index, diag_box = diag_common_area[diag_index]
                        
                        self.ques[q_index][-1].append(diag_box) # when there is a diagram which comes within the boundary of the question, the diagram that belongs to the question
                        
                        if self.ques[q_index][3][2] < diag_box[2]: self.ques[q_index][3][2] = diag_box[2] # if diagram belong to Question and outside it's boundary, extend Q-BB Boundary to the right


    def map_ques_to_bb_diag(self, obj_bb):
        '''
        Map Questions to their respective Bounding Boxes and Diagrams
        args:
            obj_bb: Object of class CreateBB which ahs all the information about bounding boxes
        '''
        image_q_index = self._map_ques_to_bbox(obj_bb)
        self._map_ques_to_diag(image_q_index, obj_bb)


class CreateBB:
    def __init__(self, cols, line_1, line_2):
        if cols > 1:
            if not line_1: raise Exception("Value of line_1 can't be None , False or 0 when there are >1 cols")
            if cols == 3:
                if not line_2: raise Exception("Value of line_2 can't be None , False or 0 when there are 3 cols")
                if line_2 < line_1: raise Exception("Value of line_2 can't be less than line_2")

        self.cols = cols
        self.line_1 = line_1
        self.line_2 = line_2

        self.table_diagrams = {}
        self.image_bb_mapping = {} # store {'original_image_name':{split_1:bboxes}, split_2:bboxes}
    

    def intersection_area(self, boxA:list, boxB:list, split:int):
        '''
        Box Intersection area: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        args:
            boxA -> Diagram BBox coors
            boxB -> Question BBox coors
            split: Split number of the question. Used in determining the right hand side upper limit
        '''
        if self.cols == 1: # can go to the maximum length of both diag and Question BB as there can't be another diagram on the right hand side
            boxB[2] = max(boxA[2], boxB[2]) 
        
        else:
            if split == 0: # can go to the max of line_1
                boxB[2] = max(self.line_1, boxB[2])
            elif split == 1: # can go to the max of line_2
                if self.line_2:
                    boxB[2] = max(self.line_2, boxB[2])
                else:
                    boxB[2] = max(boxA[2], boxB[2])
            else:
                boxB[2] = max(boxA[2], boxB[2]) # as it's the last split, it means thatthere's no boundation on the right hand side shifting

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        return  max(0, xB - xA + 1) * max(0, yB - yA + 1)
        

    def scale_boxes(self, curent_coors:list, split_num:int):
        '''
        1. If header was removed from the original image -> Add header to the ROW values so that it can scale back to original
        2. Footer shouldn't affect as it is in the end and we don't go to that value
        3. If there were splits of the image, it gets added to the COLUMN values

        args:
            current_coors: Current Coordinates which have to be scaled based on the Split  Number of ythe image
            split_num: Split of the image which currently has to be processed for scaling. Diffferent logic is applid for Split == 2 and split == 3 
        '''
        if (self.cols == 1) or (split_num == 0): return curent_coors # if the original image is used
        
        x1,y1,x2,y2 = curent_coors
        
        if split_num == 1: # means it's the second split. Add line_1 value to the x1,x2
            
            x1 += self.line_1
            x2 += self.line_1

        elif split_num == 2: # means it's the second split. Add both the values of line_1 and line_2 to x1,x2
            x1 += self.line_2 # shift horizontal
            x2 += self.line_2
        
        return [x1,y1,x2,y2]


    def _get_single_page_bboxes(self, whole_line_data:list, patterns:list, original_image_name: str, split_num:int) -> list: # for Single Image
        '''
        Given Line Data from MathPix for the Page or split, find all the bounding boxes on that page
        args:
            whole_line_data: Line data for the whole page  / split of an image
            patterns: Patterns which were found in the page HTML
            image_name: Name of the image which is being processed
            split_num: Split Numbe of the image
        '''
        if original_image_name not in self.table_diagrams: # whether there's an diagram or not, create an empty list for every new Original Image
            self.table_diagrams[original_image_name] = {}
            for _col in range(self.cols): self.table_diagrams[original_image_name][_col] = []

        if not patterns: return [] # return empty if there's no pattern detected
        if (patterns[0] is None) or (whole_line_data is None): return [[None, '']]*len(patterns) # in case line_data was missing or -DEPRECATED- due to <li> tags, patterns were sent as [None, None...]
        bboxes = []
        i = 0

        for l_index, line_data in enumerate(whole_line_data):
            coors = np.array(line_data['cnt'])
            x1 = max(0,min(coors[:,0]))
            y1 = max(0,min(coors[:,1]))
            x2 = max(coors[:,0])
            y2 = max(coors[:,1])

            x1, y1, x2, y2 = self.scale_boxes([x1,y1,x2,y2], split_num)

            if ("html" not in line_data) or (line_data['type'] in ["diagram","table"]):
                if line_data['type'] in ["diagram","table"]:
                    self.table_diagrams[original_image_name][split_num].append([x1, y1, x2, y2])
                continue
            
            if i > 0: # starting from the second value, 
                bboxes[-1][0][2] = max(x2, bboxes[-1][0][2]) # Expand to the right to cover firther lines in question
                bboxes[-1][0][0] = min(x1, bboxes[-1][0][0]) # Expand to the left to cover solution part in case it missed
        
            if i < len(patterns):        
                if (patterns[i].rstrip() in line_data['html']) or\
                    (patterns[i].replace("<br>","").replace("<div>","") in line_data['text']) or (patterns[i].replace("<br/>","").replace("</div>","") in line_data['text'])  or\
                    (patterns[i].rstrip() in line_data['text']):

                    if i > 0: # just when a new question starts, expand the boundary till the end of the last line it encountered (i.e it was the last line of answer)
                        prev_y2 = max(np.array(whole_line_data[l_index-1]['cnt'])[:,1]) # boundary of last line of the previous question's answer
                        bboxes[-1][0][3] = max(prev_y2, bboxes[-1][0][3])

                    bboxes.append(([x1,y1,x2,y2], str(uuid.uuid4())))
                    i += 1
   
        if bboxes: # last question's boundary is till the boundary of last line encountered. y2 here is always going to the coor of last line in page
            bboxes[-1][0][3] = max(y2, bboxes[-1][0][3])
        return bboxes
    

    def generate_bboxes(self,line_data:list, patterns:list, original_image_name:str, split_num:int):
        '''
        Generate bounding Boxes on a loop. It'll generate to the existing image bounding boxes or generate a fresh key
        args:
            original_image_name: In case it number of columns == 1, it is same as the image else it is the name of the original whose split is being currently processed
            split_num: it is 0 in case cols ==1 else it is the number of the split which is in process right now
        '''
        if original_image_name in self.image_bb_mapping:
            self.image_bb_mapping[original_image_name][split_num] = self._get_single_page_bboxes(line_data, patterns, original_image_name, split_num)
        else:
            self.image_bb_mapping[original_image_name] = {}
            self.image_bb_mapping[original_image_name][split_num] = self._get_single_page_bboxes(line_data, patterns, original_image_name, split_num)
