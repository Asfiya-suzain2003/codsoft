import re
import time
from colorama import Fore, Style, init

init()

def typing_effect(message):
    for char in message:
        print(char, end='',flush=True)
        time.sleep(0.03)
    print()

responses={
    r"hi|hello|hey":"hello! How can I help you today?",
    r"how are you":"I'm doing greate!Thanks for asking.",
    r"what is your name":"I'm CodBot, your virtual assistant.",
    r"(help|support)":"Sure! please tell me what you need help with.",
    r"thank you|thanks": "you're welcome!",
    r"bye|exit|quit": "Goodbye! Have a great day ahead."


}

def get_bot_response(user_input):
    user_input= user_input.lower()
    for pattern, reply in responses.items():
        if re.search(pattern, user_input):
            return reply
    return "Sorry, I didn't understand that. Could you please rephrase?"

def chatbot():
    print(Fore.CYAN+"CodBot: Hello!I am your chatbot Type'bye'to exit."+ Style.RESET_ALL)
    while True:
        user_input=input(Fore.YELLOW + "You: " +Style.RESET_ALL)
        response = get_bot_response(user_input)
        typing_effect(Fore.CYAN+"CodBot: " + response + Style.RESET_ALL)
        if re.search(r"bye|exit|quit",user_input.lower()):
            break
            

if __name__ == "__main__":
    chatbot()
    
