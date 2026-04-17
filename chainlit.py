import chainlit as cl
import subprocess
import threading
from chat import rispostarag
from chainlit.types import ThreadDict
import os
from dotenv import load_dotenv
from operator import itemgetter
import requests
import asyncio
from typing import Dict, Optional
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from database.auth import auth



load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MAX_MESSAGES = 5

@cl.password_auth_callback
def auth_callback(username: str, password: str):

    
    user=auth(username, password)
    if user:
        return cl.User(identifier=user)
    else:
        return None

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])

@cl.data_layer
def get_data_layer(): #NON TOCCARE
    return SQLAlchemyDataLayer(conninfo=DATABASE_URL)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("chat_history",[])

    for message in thread["steps"]:
        if message['type'] == 'user_message':
            cl.user_session.get("chat_history").append({"role": "user", "content": message['input']})
        elif message['type'] == 'assistant_message':
            clean_output = message['output'].split("\n\n *DEBUG*")[0]
            cl.user_session.get("chat_history").append({"role": "assistant", "content": clean_output})
    

@cl.on_message
async def on_message(message: cl.Message):
    
    chat_history = cl.user_session.get("chat_history", [])
# TODO: DECOMMENTARE PER LIMITARE I MESSAGGI
    # if len(chat_history) >= MAX_MESSAGES:
    #     await cl.ErrorMessage(content="Limite massimo di messaggi raggiunto. Inizia una nuova conversazione.").send()
    #     return
    
    user = cl.user_session.get("user")
    gruppo = user.identifier.split("-")[1] if user else "default"

    
    msg = cl.Message(content="")
    await msg.send()

    #GENERATORE
    generatore_risposta = rispostarag(message.content, gruppo, chat_history)

    
    testo_completo = ""

    
    for token in generatore_risposta:
        await msg.stream_token(token)  
        testo_completo += token        

    
    await msg.update()

    
    clean_response = testo_completo.split("\n\n *DEBUG*")[0]

    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": clean_response})

    cl.user_session.set("chat_history", chat_history)