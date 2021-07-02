import pywhatkit
import pyautogui
def text_to_hw(input_text):
        res = pywhatkit.text_to_handwriting(input_text, save_to='Handwritten_text.png')
        return res
