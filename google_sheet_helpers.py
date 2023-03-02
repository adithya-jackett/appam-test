import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import pandas as pd

from helpers import GOOGLE_CREDENTIALS_DATA, ALLOTMENT_SHEET_ID, ALLOTMENT_WORKSHEET_NAME, TEACHER_NAME_SHEET_ID, \
    TEACHER_NAME_WORKSHEET, print_with_date

# Set up the credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(GOOGLE_CREDENTIALS_DATA), scope)

# Connect to the allotment sheet
gc = gspread.authorize(credentials)
sh = gc.open_by_key(ALLOTMENT_SHEET_ID)
worksheet = sh.worksheet(ALLOTMENT_WORKSHEET_NAME)
print_with_date("updating worksheet to default")

# Connect to the teacher name sheet
teacherSheet = gc.open_by_key(TEACHER_NAME_SHEET_ID)
teacherWorksheet = teacherSheet.worksheet(TEACHER_NAME_WORKSHEET)


# TODO: Hide the side panel while the app is running
# TODO: Error images need to added
# TODO: Implement the flow for the MCQ Flow
# TODO: Cleanup required for the unused functions

# Preparing DF to append to the google sheet
def prepare_df_with_chapters(flattenedLabels, **metadataDict):
    data = {"Class": [], "Curriculum": [], "Book/Source": [], "Subject": [], "Chapter": [], "user-ID": [],
            "Teacher Name": [], "NumQs by Model": []}
    metadata = {i:[] for i in metadataDict.keys()}
    for key in flattenedLabels["Chapter"]:
        # Adding just the chapter
        data['Class'].append("")
        data['Curriculum'].append("")
        data['Book/Source'].append("")
        data['Subject'].append("")
        data['Chapter'].append(key)
        data['user-ID'].append("")
        data['Teacher Name'].append("")
        data['NumQs by Model'].append("")
        {metadata[i].append("") for i in metadataDict.keys()}
        for sourceImageIds in flattenedLabels["Chapter"][key]:
            # Adding the source image ids from the chapter
            data['Class'].append(flattenedLabels["Class"])
            data['Curriculum'].append(flattenedLabels["Curriculum"])
            data['Book/Source'].append(flattenedLabels["Book/Source"])
            data['Subject'].append(flattenedLabels["Subject"])
            data['Chapter'].append(sourceImageIds)
            data['user-ID'].append(flattenedLabels["user-ID"])
            data['Teacher Name'].append(flattenedLabels["Teacher Name"])
            data['NumQs by Model'].append(flattenedLabels["Chapter"][key][sourceImageIds])
            {metadata[i].append(metadataDict[i]) for i in metadataDict.keys()}
    data.update(metadata)
    preparedDf = pd.DataFrame(data)
    return preparedDf


# Append the required data to the Google sheet with chapters
def append_image_ids_to_google_sheet_with_chapters(flattenedLabels, **metadataDict):
    rowsToAdd = prepare_df_with_chapters(flattenedLabels, **metadataDict)
    worksheet.append_rows(rowsToAdd.values.tolist())
    return


# Flatten labels when multiple chapters exist
def flatten_labels_multiple_chapters(labels, chapterAndImageIDs, username, teacherName):
    # Checking for string index out of range exception
    if len(labels["classes"]) > 0:
        flatClass = labels["classes"][0]
    else:
        flatClass = ""

    if len(labels["source"]) > 0:
        flatSource = labels["source"][0]
    else:
        flatSource = ""

    # Actually flattening the labels
    flattenedLabels = {
        "Class": flatClass,
        "Curriculum": labels["curriculum"],
        "Book/Source": flatSource,
        "Subject": labels["subject"],
        "Chapter": chapterAndImageIDs,
        "user-ID": username,
        "Teacher Name": teacherName
    }
    return flattenedLabels


# Get the predefined teacher name list in the form of a tuple
def get_teacher_name_tuple():
    teacherDf = pd.DataFrame(teacherWorksheet.get_all_records())
    nonEmptyVal = list(filter(None, teacherDf["Customer Name"].values))
    return nonEmptyVal


# Get the predefined digitiser name list in the form of a tuple
def get_digitiser_name_tuple():
    digitiserDf = pd.DataFrame(teacherWorksheet.get_all_records())
    nonEmptyVal = list(filter(None, digitiserDf["Digitised By"].values))
    return nonEmptyVal


# Get metadata export worksheet names
def get_export_worksheet_names():
    worksheetDf = pd.DataFrame(teacherWorksheet.get_all_records())
    nonEmptyVal = list(filter(None, worksheetDf["Export Worksheet Name"].values))
    return nonEmptyVal


# Update the worksheet to export to, dynamically
def update_export_sheet(newWorksheetName):
    global ALLOTMENT_WORKSHEET_NAME, worksheet
    ALLOTMENT_WORKSHEET_NAME = newWorksheetName
    worksheet = sh.worksheet(ALLOTMENT_WORKSHEET_NAME)


# Get allotment worksheet url
def get_allotment_worksheet_url():
    return worksheet.url


# Get Teacher Name Worksheet url
def get_teacher_name_worksheet_url():
    return teacherWorksheet.url
