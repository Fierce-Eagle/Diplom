from tkinter import *
from interface import Application

if __name__ == '__main__':
    window = Tk()
    window.title("Название приложения")
    window.geometry('1300x800')
    window.resizable(False, False)
    app = Application(window)

    window.mainloop()

