import json
import os

import requests
import streamlit as st
from helpers import JACKETT_WEBSERVICE_URL
from pandas import DataFrame as DF

# TODO: For different question types, there should be a different mapping data? Maybe not, because even Manu doesn't do so
# TODO: Does not work with suggested labelling yet - since that will create multiple mapping data

LOGIN_ENDPOINT = st.secrets["LOGIN_ENDPOINT"]
CREATE_MAPPING_DATA_ENDPOINT = st.secrets["CREATE_MAPPING_DATA_ENDPOINT"]
loginHeaders = {"accept": "application/json", "content-type": "application/json"}
loginData = {"username": "rachiket", "password": "rachiket"}
questionDifficultyStatic = "Medium"
bloomTaxonomyStatic = "Remebering"
excelFileName = "mappingData.xlsx"
usernameStatic = "jackett_intern"

# Get the admin login token from Jackett Webservice
def get_jackett_auth_token():

    respLogin = requests.post(url=JACKETT_WEBSERVICE_URL+LOGIN_ENDPOINT, headers=(loginHeaders), data=json.dumps(loginData))
    respLoginJson = respLogin.json()

    authToken = respLoginJson['data']['token']

    return str(authToken)


# Function to create a Mapping Data ready XLSX file out of a list
def create_mapping_data_excel(mappingDataItems, questionType):
    excelData = {"curriculum": [], "classes": [], "subjectTags": [], "chapter": [], "questionDifficulty": [], "questionType": [], "sources": [], "bloomTaxonomies": [], "username": []}
    for chapter in mappingDataItems["Chapter"]:
        excelData['classes'].append(mappingDataItems["Class"])
        excelData['curriculum'].append(mappingDataItems["Curriculum"])
        excelData['sources'].append(mappingDataItems["Book/Source"])
        excelData['subjectTags'].append(mappingDataItems["Subject"])
        excelData['chapter'].append(chapter)
        excelData['questionDifficulty'].append(questionDifficultyStatic)
        excelData['questionType'].append(questionType)
        excelData['bloomTaxonomies'].append(bloomTaxonomyStatic)
        excelData['username'].append(usernameStatic)

    # convert into dataframe
    df = DF(data=excelData)

    # convert into excel
    df.to_excel(excelFileName, index=False)

    return excelData


# Call backend API to create the mapping data for the user
def create_mapping_data_jackett_webservice(mappingDataItems, questionType):
    create_mapping_data_excel(mappingDataItems, questionType)
    mappingDataHeaders = {"accept": "application/json", "Authorization": "Bearer " + get_jackett_auth_token()}
    mappingResponse = requests.post(url=JACKETT_WEBSERVICE_URL+CREATE_MAPPING_DATA_ENDPOINT, headers=(mappingDataHeaders), files={'file': (excelFileName, open(excelFileName, 'rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})

    if os.path.exists(excelFileName):  # remove file if exists
        os.remove(excelFileName)

    return mappingResponse



# TEST DATA
# mitem = {
#     "Class": "12",
#     "Curriculum": "MyCurr",
#     "Book/Source": "MySource",
#     "Subject": "Test Subject",
#     "Chapter": "Test Ch",
#     "user-ID": "rachiket+12@tryjackett.com"
# }
# t1 = create_mapping_data_jackett_webservice(mitem, "MCQ")
# print_with_date(t1)
