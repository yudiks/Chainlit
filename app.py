# import chainlit as cl
from chainlit.types import ThreadDict

# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import StrOutputParser
# from langchain.chains import LLMChain

from tempfile import NamedTemporaryFile
from typing import List
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.chains import LLMChain

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_file(*, file: AskFileResponse) -> List[Document]:
    """Processes one PDF file from a Chainlit AskFileResponse object by first
    loading the PDF document and then chunk it into sub documents. Only
    supports PDF files.

    Args:
        file (AskFileResponse): input file to be processed

    Raises:
        ValueError: when we fail to process PDF files. We consider PDF file
        processing failure when there's no text returned. For example, PDFs
        with only image contents, corrupted PDFs, etc.

    Returns:
        List[Document]: List of Document(s). Each individual document has two
        fields: page_content(string) and metadata(dict).
    """
    # We only support PDF as input.
    if file.type != "application/pdf":
        raise TypeError("Only PDF files are supported")

    with NamedTemporaryFile() as tempfile:
        print(file)
        # tempfile.write(file.content)

        # with open(file.path, "r", encoding="utf-8") as f:
        #     text = f.read()
        #     tempfile.write(text)

        ######################################################################
        # Exercise 1a:
        # We have the input PDF file saved as a temporary file. The name of
        # the file is 'tempfile.name'. Please use one of the PDF loaders in
        # Langchain to load the file.
        # NOTE: https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#using-pdfplumber
        ######################################################################
        loader = PDFPlumberLoader(file.path)
        documents = loader.load()
        ######################################################################

        ######################################################################
        # Exercise 1b:
        # We can now chunk the documents now it is loaded. Langchain provides
        # a list of helpful text splitters. Please use one of the splitters
        # to chunk the file.
        # NOTE: https://python.langchain.com/docs/modules/data_connection/text_splitter#using-recursivecharactertextsplitter
        ######################################################################
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        
        # ######################################################################

        # We are adding source_id into the metadata here to denote which
        # source document it is.
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        if not docs:
            raise ValueError("PDF file parsing failed.")

        return docs


@cl.on_chat_start
async def on_chat_start():
    ######################################################################
    # Exercise 1c:
    # At the start of our Chat with PDF app, we will first ask users to
    # upload the PDF file they want to ask questions against.
    # 
    # Please use Chainlit's AskFileMessage and get the file from users.
    # Note for this course, we only want to deal with one single file.
    ######################################################################
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please Upload the PDF file you want to chat with...",
            accept=["application/pdf"],
            max_size_mb=20,
        ).send()
    file = files[0]
    ######################################################################

    # Send message to user to let them know we are processing the file
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    msg.content = f"`{file.name}` processed. Loading ..."
    await msg.update()

    model =  ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
        max_tokens=500,
        model="TheBloke/una-cybertron-7B-v2-GGUF",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Chainlit GPT, a helpful assistant.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())

    # We are saving the chain in user_session, so we do not have to rebuild
    # it every single time.
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    # Let's load the chain from user_session
    chain = cl.user_session.get("chain")  # type: LLMChain
    response = await chain.arun(
        question=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=response).send()    


@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")

@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")