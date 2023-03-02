# ENV="PROD" streamlit run app.py --server.maxUploadSize 2048 --server.port 1234 --logger.level=info 2 > streamlit_logs.log

from helpers import *
from app_helpers import *
import pytesseract
from pytesseract import Output


test_tesseract_image = cv2.imread('tesseractr_test_image.jpg')
test_tesseract_data = pytesseract.image_to_string(test_tesseract_image, lang='eng', config='--psm 6')
st.info(f"Tesseract Loaded and output from test is ::  {str(test_tesseract_data)}", icon="ℹ️")

st.set_page_config(page_title="Question Digitisation",page_icon="📖", layout = "wide")

if ENV == "DEV": st.warning("App running on Development mode",icon="⚠️")

init_states()

st.markdown("""#### Upload either a <span style="color:red">Single PDF File</span> or a <span style="color:green">ZIP file which has images</span>""", 
            unsafe_allow_html=True)

st.file_uploader("Upload File", type = ["pdf", "zip"], label_visibility="hidden", key = 'uploaded')

if st.session_state['uploaded'] is not None:

    if st.session_state['uploaded'].name.endswith(".zip") or (st.session_state['show_second_flow']):
        
        if st.session_state['uploaded'].name.endswith(".zip"): # if zip has been uploaded
            if st.session_state['image_file_paths'] is None:
                with st.spinner("Extracting Zip file..."):
                    process_zip(st.session_state['uploaded'])

        create_sidebar() # all the Sidebar Functiionality like Adding Username, Password, Tags, Tag File Upload
        second_flow_run_pipelines() # show Image, Header Footer + Columns, Run Pipleine Button
        

    else: # it it's a PDF file

        extract = st.session_state['file_dir'] is None # whether to show Extract or show zip, download buttons
        if extract:
            with st.spinner("Generating Images from PDF....."):
                process_pdf()
                st.experimental_rerun()
        
        else:
            if st.session_state["col_detection"] is None:
                with st.spinner("Collecting Single / Multi Column Data..."):
                    st.session_state["col_detection"] = get_cols_data(st.session_state['image_file_paths'])
            st.warning(f"Single / Multi Column stats:  {st.session_state['col_detection']}", icon="🚩")

            st.markdown("""<h5> Continue or download <code style="color:green">zip</code> file of images</h3>""",unsafe_allow_html=True)
            c1, c2, _ = st.columns([2,2,8])
            with c1:
                if st.button("Continue without Downloading"):
                    st.session_state['show_second_flow'] = True # Show second flow without downloading zip
                    st.experimental_rerun()
            
            with c2:
                with open(st.session_state['zipped_path'], 'rb') as file_data:
                    st.download_button('Download Zip', file_data, file_name = st.session_state['zipped_path'].split("/")[-1])