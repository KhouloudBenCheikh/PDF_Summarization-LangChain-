import streamlit as st 
import os

#Main function to run the UI Interface
def main():
    st.set_page_config(page_title=" PDf Summarizer")
    st.title('PDF Summarizing_ Langchain 🦜')
    st.write("Summarize your pdf files here")
    st.divider()

    pdf = st.file_uploader('upload your Pdf Document', type='pdf')

    #button
    submit = st.button("Generate summary")


if __name__ == '__main__':
    main()