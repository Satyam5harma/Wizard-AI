import speech_recognition as sr
import webbrowser
import pyttsx3
import requests
import musicLibrary
from openai import OpenAI
from gtts import gTTS
import pygame
import os

# pip install pocketsphinx

recognizer = sr.Recognizer()
engine = pyttsx3.init() 
newsapi = "Your-API-KEY" #I CAN'T SHOW YOU MY API KEY, PLEASE GENERATE AND USE YOUR OWN KEY.

def speak_old(text):
    engine.say(text)
    engine.runAndWait()

def speak(text):
    tts = gTTS(text)
    tts.save('temp.mp3') 

    # Initialize Pygame mixer
    pygame.mixer.init()

    # Load the MP3 file
    pygame.mixer.music.load('temp.mp3')

    # Play the MP3 file
    pygame.mixer.music.play()

    # Keep the program running until the music stops playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    pygame.mixer.music.unload()
    os.remove("temp.mp3") 

def Process(command):
    print(command)
    

def processCommand(c):
    if "open google" in c.lower():
        webbrowser.open("https://google.com")
    elif "open facebook" in c.lower():
        webbrowser.open("https://facebook.com")
    elif "open youtube" in c.lower():
        webbrowser.open("https://youtube.com")
    elif "open linkedin" in c.lower():
        webbrowser.open("https://linkedin.com")

    elif c.lower().startswith("play"):
        song = c.lower().split(" ")[1]
        link = musicLibrary.music[song]
        webbrowser.open(link)

    elif "news" in c.lower():
      r = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi}")
    
    if r.status_code == 200:
        data = r.json()
        articles = data.get('articles', [])
        
        if articles:
            speak("Here are the latest headlines.")
            for i, article in enumerate(articles[:5]):  # Limit to top 5 headlines
                speak(f"Headline {i + 1}: {article['title']}")
        else:
            speak("Sorry, I couldn't find any news at the moment.")
    else:
        speak("Failed to fetch news. Please check your API key or internet connection.")




if __name__ == "__main__":
    speak("Starting-up Wizard....")
    while True:
        # Listen for the wake word "wizard"
        # obtain audio from the microphone
        r = sr.Recognizer()
         
        print("recognizing...")
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = r.listen(source, timeout=2, phrase_time_limit=5)
            word = r.recognize_google(audio)
            if(word.lower() == "wizard"):
                speak("yes sir!")
                # Listen for command
                with sr.Microphone() as source:
                    print("Wizard Active...")
                    audio = r.listen(source)
                    command = r.recognize_google(audio)

                    processCommand(command)


        except Exception as e:
            print("Error; {0}".format(e))