from tkinter import *
from tkinter import filedialog
import torch 
from torchvision import transforms
from PIL import Image, ImageTk
from models import Encoder, Decoder, Model
import numpy as np 
import sqlite3


def main():
    root = Tk()
    conn = sqlite3.connect('Hospital_image.db')
    cur = conn.cursor()

    
    root.title("Enhance")
    root.geometry(f'700x700+0+0')

    load = Image.open('WhatsApp Image 2020-12-07 at 5.45.08 PM.jpeg')
    image = ImageTk.PhotoImage(load)

    img = Label(root, image=image)
    img.place(x=0, y=0, relheight=1, relwidth=1)


    def search_db():

        record_win = Tk()
        record_win.geometry("450x150")

        record_win.title("Records")
        canvas = Canvas(record_win, height=155, width=450, bg="#ffcc66")
        canvas.pack()

        search_id = Label(canvas, text="Patient ID", font=["Times", 15, "bold"], bg="white")
        search_id.place(x=10, y=10)

        search_input = Entry(canvas, font=["Verenda", 15])
        search_input.place(x=5, y=40)

        def search_data():
            id = search_input.get()
            search_input.delete(0, END)

            cur.execute(f"SELECT * FROM Hospital_image WHERE ID = {id}")
            
            data = cur.fetchall()
            name, id, model_name_from_db, img = data[0]
     
            name_from_db = Label(canvas, text=f'{name}', font=["Verenda", 15, "bold"], bg="white")
            name_from_db.place(x=250, y=40)

            img_type = Label(canvas, text=f'{model_name_from_db}', font=["Verenda", 15, "bold"], bg="white")
            img_type.place(x=270, y=70)
            

            if model_name_from_db == "mri" or model_name_from_db == "xray":
                img_from_db = Image.frombuffer(mode="L", data=img, size=(128,128))

                img_from_db.show()


        search_button = Button(record_win, text="Search", bg="white", font=["Verenda", 15], command=search_data)
        search_button.place(x=5, y=70)

        record_win.mainloop()
    
    def database(name, id, img_obj, model_from_db):
        query = '''CREATE TABLE IF NOT EXISTS Hospital_image(Name TEXT, ID TEXT, TYPE TEXT, IMAGE BLOB)'''
        cur.execute(query)
        img_np = np.asarray(img_obj)
        buffer = img_np.tostring()

        cur.execute("INSERT INTO Hospital_image VALUES(?, ?, ?, ?)",(name, id, model_from_db, buffer)) 
        conn.commit()


    def upload():
        filename = filedialog.askopenfilename(initialdir='/', title="Select an Image", filetypes=(("files jpg", "*.jpg"), ("all files", "*.*")))
        img = Image.open(filename)
        img = img.resize((128,128))
        render = ImageTk.PhotoImage(img)
        
        img_before = Label(root, image=render)
        
        img_before.image = render
        img_before.place(x=450, y=250)
        
        model_name = entry_box_model.get()
        model_name = model_name.lower()

        if model_name == 'mri':
            encoder = Encoder(in_channel=1, out_channel=64)
            decoder = Decoder(in_channel=192, out_channel=128, final_out=1)
            model = Model(encoder, decoder)

            trans = transforms.Compose([transforms.Resize((128,128)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])

            model.load_state_dict(torch.load('de_MRI.pt', map_location=torch.device('cpu')))

        elif model_name == 'xray':
            encoder = Encoder(in_channel=1, out_channel=32)
            decoder = Decoder(in_channel=96, out_channel=128, final_out=1)
            model = Model(encoder, decoder)

            trans = transforms.Compose([transforms.Resize((128,128)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])

            model.load_state_dict(torch.load('model_xray.pt', map_location=torch.device('cpu')))

        else:
            encoder = Encoder(in_channel=3, out_channel=32)
            decoder = Decoder(in_channel=96, out_channel=128, final_out=3)
            model = Model(encoder, decoder)
            
            trans = transforms.Compose([transforms.Resize((128,128)),
                                        transforms.ToTensor()])

            model.load_state_dict(torch.load('deblur_cell.pt', map_location=torch.device('cpu')))

        img_to_model = Image.open(filename)
        img_to_model = trans(img_to_model)
        from_model = model(img_to_model.unsqueeze(0))

        from_model = transforms.ToPILImage()(from_model.squeeze(0))
        render = ImageTk.PhotoImage(from_model)
        img_after = Label(root, image=render)
        
        img_after.image = render
        img_after.place(x=450, y=450)

        def save_db():
            name = entry_box_name.get()
            id = entry_box_id.get()
            model_type = entry_box_model.get()
            img = from_model
            database(name, id, img, model_type)

            entry_box_id.delete(0, END)
            entry_box_name.delete(0, END)
            entry_box_model.delete(0, END)
            img_before.destroy()
            img_after.destroy()

        def del_img():
            img_before.destroy()
            img_after.destroy()
            entry_box_id.delete(0, END)
            entry_box_name.delete(0, END)
            entry_box_model.delete(0, END)
        
        
        button_to_db = Button(root, text="save", command=save_db, font=["Verneda", 15], bg="white")
        button_to_db.place(x=280, y=650)
        
        clear_image_input = Button(root, text="Clear Inputs", font=["Verneda", 15], bg="white", command=del_img)
        clear_image_input.place(x = 430, y = 600)
        

    name_label = Label(root, text="Patient Name", font=["Verenda", 15], bg="white")
    name_label.place(x=5, y=200)
    entry_box_name = Entry(root, font=["Verenda", 15])
    entry_box_name.place(x=150, y=200)

    id_label = Label(root, text="Patient ID", font=["Verneda", 15], bg="white")
    id_label.place(x=5, y=250)
    entry_box_id = Entry(root, font=["Verenda", 15])
    entry_box_id.place(x=150, y=250)

    model_type = Label(root, text="Image Type", font=["Verneda", 15], bg="white")
    model_type.place(x=5, y=300)
    entry_box_model = Entry(root, font=["Verenda", 15])
    entry_box_model.place(x=150, y=300)



    button_upload = Button(root, text="Upload Image", command=upload, font=["Verneda", 15], bg="white")
    button_upload.place(x=450, y=200)

    predict = Label(root, text="Clear", font=["Verneda", 15], bg="white")
    predict.place(x=500, y=400)

    show_recods = Button(root, text="Search Records", font=["Verneda", 25], bg="white", command=search_db)
    show_recods.place(x=50, y=90)

    root.mainloop()

def login():
    login_root = Tk()
    login_root.geometry("700x700+150+50")

    login_root.title("Login")
    load = Image.open('Webp.net-resizeimage (1).jpg')
    image = ImageTk.PhotoImage(load)
    
    img = Label(login_root, image=image)
    img.place(x=0, y=0, relheight=1, relwidth=1)
    
    id = Label(login_root, text="Enter ID", font=["Helvetica", 25, "bold"], bg="white")
    id.place(x=250, y=350)
    id_entry = Entry(login_root, font=["Helvetica", 25 ])
    id_entry.place(x=160, y=400)
    

    def show():
        if id_entry.get() == "0000":  ## default id
            login_root.destroy()
            main()

    password_button = Button(login_root, text="Login", command=show, font=["Helvetica", 25], bg="white")
    password_button.place(x=270, y=600)   
                   

    login_root.mainloop()


login()     