import chainlit as cl
from chainlit.types import ThreadDict

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain

@cl.on_message
async def main(message: cl.Message):
    await cl.Message(content=f"Received: {message.content}").send()

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(
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
            (
                "human",
                "{question}"
            ),
        ]
    )
    chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())
    # We are saving the chain in user_session, so we do not have to rebuild
    # it every single time.
    cl.user_session.set("chain", chain)

    print("A new chat session has started!")

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