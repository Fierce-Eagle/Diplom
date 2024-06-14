from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from joblib import load

from model import *
from my_dataset import *
from train import test_model


class Application:
    def __init__(self, window):
        """
        Параметры
        """
        self.window = window

        self.y_pred_test = None
        self.filename = None
        self.test_model_directory = None
        self.model = None
        self.scaler = None
        self.test_dataset = None

        self.signal_1_list = []
        self.signal_2_list = []
        self.signal_3_list = []
        self.spectr_1_list = []
        self.spectr_2_list = []
        self.spectr_3_list = []
        self.spectrum_list = []

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        """
        Виджеты
        """
        # Места для графиков
        self.canv_signal_1 = Canvas(window, width=250, height=200, bg="white")
        self.canv_signal_1.place(x=30, y=70)
        self.canv_signal_2 = Canvas(window, width=250, height=200, bg="white")
        self.canv_signal_2.place(x=300, y=70)
        self.canv_signal_3 = Canvas(window, width=250, height=200, bg="white")
        self.canv_signal_3.place(x=580, y=70)

        self.canv_spectr_1 = Canvas(window, width=250, height=200, bg="white")
        self.canv_spectr_1.place(x=30, y=320)
        self.canv_spectr_2 = Canvas(window, width=250, height=200, bg="white")
        self.canv_spectr_2.place(x=300, y=320)
        self.canv_spectr_3 = Canvas(window, width=250, height=200, bg="white")
        self.canv_spectr_3.place(x=580, y=320)

        self.canv_spectrum = Canvas(window, width=800, height=200, bg="white")
        self.canv_spectrum.place(x=30, y=570)

        # Комбобокс
        self.combo_box_items = Combobox(window)
        self.combo_box_items['state'] = 'readonly'
        self.combo_box_items.place(x=850, y=210)

        # Кнопки
        self.btn_load_model = Button(window, text="Загрузить модель", command=self.load_model_button_click)
        self.btn_load_model.place(x=1160, y=73)
        self.btn_load_data = Button(window, text="Загрузить данные", command=self.load_data_button_click)
        self.btn_load_data.place(x=1160, y=118)
        self.btn_test_model = Button(window, text="Протестировать модель", command=self.test_model_button_click)
        self.btn_test_model.place(x=1090, y=158)

        # Окошка для ввода/вывода
        self.input_box_load_model = Entry(master=window, width=50)
        self.input_box_load_model['state'] = 'readonly'
        self.input_box_load_model.place(x=850, y=75)
        self.input_box_load_data = Entry(master=window, width=50)
        self.input_box_load_data['state'] = 'readonly'
        self.input_box_load_data.place(x=850, y=120)
        self.input_box_show_acc = Entry(master=window, width=15)
        self.input_box_show_acc['state'] = 'readonly'
        self.input_box_show_acc.place(x=990, y=160)

        # Label
        self.frame_type_label = Label(master=window, text=' ')
        self.frame_type_label.place(x=1020, y=210)
        # Unnamed label
        Label(master=window, text='Загрязненность:', font=18).place(x=850, y=160)
        Label(master=window, text='Сигналы:', font=48).place(x=30, y=40)
        Label(master=window, text='Спектры:', font=48).place(x=30, y=290)
        Label(master=window, text='Фрейм для проверки:', font=48).place(x=30, y=540)

    """
    Функции
    """
    def load_model_button_click(self):
        self.filename = filedialog.askdirectory()

        self.model = Perceptron_fc3(192 * 6, 2)
        self.model.load_state_dict(torch.load(self.filename + '/model_fc3_0.pth'))
        self.model.eval()
        self.scaler = load(self.filename + '/scaler_fc3_0.bin')
        self.input_box_load_model['state'] = 'normal'
        self.input_box_load_model.delete(0, 'end')
        self.input_box_load_model.insert(0, str(self.filename))
        self.input_box_load_model['state'] = 'readonly'

    def load_data_button_click(self):
        self.test_model_directory = filedialog.askdirectory()
        if self.test_model_directory is None:
            return
        self.input_box_load_data['state'] = 'normal'
        self.input_box_load_data.delete(0, 'end')
        self.input_box_load_data.insert(0, str(self.test_model_directory))
        self.input_box_load_data['state'] = 'readonly'
        test_dataset, signal_1, signal_2, signal_3, spectr_1, spectr_2, spectr_3, spectrums = create_test_dataset(
                                                                         folder_path=self.test_model_directory,
                                                                         spectrum_length=192,
                                                                         time_for_frame=0.04,
                                                                         smoothing_window_size=8,
                                                                         frame_stride_percent=0,
                                                                         scaler=self.scaler)
        self.test_dataset = test_dataset
        self.signal_1_list = signal_1
        self.signal_2_list = signal_2
        self.signal_3_list = signal_3
        self.spectr_1_list = spectr_1
        self.spectr_2_list = spectr_2
        self.spectr_3_list = spectr_3
        self.spectrum_list = spectrums
        items = [i for i in range(len(spectrums))]
        self.combo_box_items['values'] = items
        self.combo_box_items.current(0)
        self.combo_box_items.bind("<<ComboboxSelected>>", self.selected)
        self.frame_type_label.config(text=' ', foreground='black')
        self.plot(0)

    def test_model_button_click(self):
        self.y_pred_test = test_model(model=self.model, loader_test=self.test_dataset, device=self.device)
        accuracy = self.y_pred_test.tolist().count(1) / len(self.y_pred_test) * 100
        self.input_box_show_acc['state'] = 'normal'
        self.input_box_show_acc.delete(0, 'end')
        accuracy_str = '%.4f ' % accuracy
        accuracy_str += '%'
        self.input_box_show_acc.insert(0, accuracy_str)
        self.input_box_show_acc['state'] = 'readonly'

        if self.y_pred_test[0] == 1:
            self.frame_type_label.config(text='Грязный', foreground='red')
        else:
            self.frame_type_label.config(text='Чистый', foreground='green')

    def selected(self, event):
        index = int(event.widget.get())
        if self.y_pred_test is not None:
            if self.y_pred_test[index] == 1:
                self.frame_type_label.config(text='Грязный', foreground='red')
            else:
                self.frame_type_label.config(text='Чистый', foreground='green')
        self.plot(index)

    def plot(self, index):
        fig1 = Figure(figsize=(5, 4), dpi=50)
        plot1 = fig1.add_subplot()
        fig1.subplots_adjust(left=0.1, right=0.97, top=0.99, bottom=0.07)
        plot1.plot(self.signal_1_list[index], color='red')
        self.canv_signal_1 = FigureCanvasTkAgg(fig1, master=self.window)
        self.canv_signal_1.draw()
        self.canv_signal_1.get_tk_widget().place(x=30, y=70)

        fig2 = Figure(figsize=(5, 4), dpi=50)
        plot2 = fig2.add_subplot()
        fig2.subplots_adjust(left=0.1, right=0.97, top=0.99, bottom=0.07)
        plot2.plot(self.signal_2_list[index], color='green')
        self.canv_signal_2 = FigureCanvasTkAgg(fig2, master=self.window)
        self.canv_signal_2.draw()
        self.canv_signal_2.get_tk_widget().place(x=300, y=70)

        fig3 = Figure(figsize=(5, 4), dpi=50)
        plot3 = fig3.add_subplot()
        fig3.subplots_adjust(left=0.1, right=0.97, top=0.99, bottom=0.07)
        plot3.plot(self.signal_3_list[index])
        self.canv_signal_3 = FigureCanvasTkAgg(fig3, master=self.window)
        self.canv_signal_3.draw()
        self.canv_signal_3.get_tk_widget().place(x=580, y=70)

        fig4 = Figure(figsize=(5, 4), dpi=50)
        plot4 = fig4.add_subplot()
        fig4.subplots_adjust(left=0.07, right=0.97, top=0.99, bottom=0.07)
        plot4.plot(self.spectr_1_list[index], color='red')
        self.canv_spectr_1 = FigureCanvasTkAgg(fig4, master=self.window)
        self.canv_spectr_1.draw()
        self.canv_spectr_1.get_tk_widget().place(x=30, y=320)

        fig5 = Figure(figsize=(5, 4), dpi=50)
        plot5 = fig5.add_subplot()
        fig5.subplots_adjust(left=0.07, right=0.97, top=0.99, bottom=0.07)
        plot5.plot(self.spectr_2_list[index], color='green')
        self.canv_spectr_2 = FigureCanvasTkAgg(fig5, master=self.window)
        self.canv_spectr_2.draw()
        self.canv_spectr_2.get_tk_widget().place(x=300, y=320)

        fig6 = Figure(figsize=(5, 4), dpi=50)
        plot6 = fig6.add_subplot()
        fig6.subplots_adjust(left=0.07, right=0.97, top=0.99, bottom=0.07)
        plot6.plot(self.spectr_3_list[index])
        self.canv_spectr_3 = FigureCanvasTkAgg(fig6, master=self.window)
        self.canv_spectr_3.draw()
        self.canv_spectr_3.get_tk_widget().place(x=580, y=320)

        fig7 = Figure(figsize=(16, 4), dpi=50)
        plot7 = fig7.add_subplot()
        fig7.subplots_adjust(left=0.03, right=0.98, top=0.99, bottom=0.1)
        plot7.plot(self.spectrum_list[index], color='black')
        self.canv_spectrum = FigureCanvasTkAgg(fig7, master=self.window)
        self.canv_spectrum.draw()
        self.canv_spectrum.get_tk_widget().place(x=30, y=570)