import uuid
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import json
from tkinter import *
import copy
import os.path
import pathlib



def j_tree(tree, parent, dic, blacklist = None, whitelist = None, start_open = False):
    global if_lst, path_lst
    # If any sub keys exisit
    for key in sorted(dic.keys()):
        #print("")
        #print(key)
        #print(blacklist)
        #print(whitelist)
        if((blacklist != None and key in blacklist) or (whitelist != None and key not in whitelist)):
            continue

        uid = uuid.uuid4()
        if isinstance(dic[key], dict):
            tree.insert(parent, 'end', uid, text=key, open=start_open)
            j_tree(tree, uid, dic[key])
        elif isinstance(dic[key], tuple):
            tree.insert(parent, 'end', uid, text=str(key) + '()', open=start_open)
            j_tree(tree, uid,
                   dict([(i, x) for i, x in enumerate(dic[key])]))
        elif isinstance(dic[key], list):
            tree.insert(parent, 'end', uid, text=key, open=start_open)
            if_lst.append(str(uid))
            j_tree(tree, uid,
                   dict([(i, x) for i, x in enumerate(dic[key])]))
        # Different data type should be handled differently
        else:
            value = dic[key]
            
            if isinstance(value, str):
                # Use "/" instead of "\"
                if "\\" in value or "/" in value:
                    p = pathlib.PureWindowsPath(value)
                    value = p.as_posix()
                    path_lst.append(str(uid))
                tree.insert(parent, 'end', uid, text=key, value=str(value), open=start_open)
            elif isinstance(value, bool):
                if value == 1:
                    tree.insert(parent, 'end', uid, text=key, value="true", open=start_open)
                else:
                    tree.insert(parent, 'end', uid, text=key, value="false", open=start_open)
            else:
                tree.insert(parent, 'end', uid, text=key, value=value, open=start_open)

            
 
class JsonTreeView:
    edit = True

    def __init__(self, jfullpath = r"C:\WormWatcher\config.json", frame = None, whitelist=None, blacklist=None, start_open = False, *args, **kwargs):
        self.datapath = jfullpath
        if not os.path.isfile(self.datapath):
            messagebox.showerror("ERROR", self.datapath+" is an invalid path!")
            return
        self.tr_frame = frame
        self.whitelist = whitelist
        self.blacklist = blacklist
        self.start_open = start_open
        self.tk_tree_view(*args)
        

    def tk_tree_view(self, *args, **kwargs):
        global click_num, if_lst, path_lst

        click_num = 0
        if_lst = []
        self.lst_elem = []
        path_lst = []
        with open(self.datapath, "r") as myfile:
                    self.data = json.load(myfile)
        # Provided frame to put in
        if self.tr_frame:
            tree_frame = self.tr_frame
            self.root = None
        else:
            # Setup the self.root UI
            self.root = tk.Tk()
            self.root.title("tk_tree_view")
            self.root.geometry("530x600")
            self.root.resizable(0, 0)
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            # Setup the Frames
            tree_frame = ttk.Frame(self.root, padding="3")
            tree_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # Setup the Tree
        tree_frame.update()
        self.fr_width = tree_frame.winfo_width() 
        self.val_width = int(self.fr_width/2.1)
        self.text_width = self.fr_width - self.val_width
        #style = ttk.Style()
        #style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Helvetica', 9)) # Modify the font of the body

        self.tree = ttk.Treeview(tree_frame)
        self.tree["columns"]=("one")
        self.tree.column("#0", width=self.text_width, stretch=tk.NO)
        self.tree.column("one", width=self.val_width, minwidth=self.val_width, anchor=tk.CENTER)
        self.tree.heading("#0",text=" ",anchor=tk.W)
        self.tree.heading("one", text="Values",anchor=tk.CENTER)
        
        verscrlbar = ttk.Scrollbar(tree_frame, orient ="vertical", command = self.des_wid) 
        verscrlbar.pack(side ='right', fill ='y')

        self.tree.configure(yscrollcommand = verscrlbar.set) 
        verscrlbar.update()
        sc_bar_width = verscrlbar.winfo_width()
        #self.tree.column('Values', width=self.val_width, anchor='center')
        #self.tree.heading('Values', text='Values')
        j_tree(self.tree, '', self.data, self.blacklist, self.whitelist, self.start_open)
        self.tree.pack(side ='right', fill=tk.BOTH, expand=1)


        # Elements in lists
        for i in if_lst:
            child_iid = self.tree.get_children(i)
            for j in child_iid:
                self.lst_elem.append(j)

        self.tree.bind('<Double-1>', self.onDoubleClick) # Double-click the left button to enter the edit
        self.tree.bind('<ButtonRelease-1>', self.onSingleClick_list)
        self.tree.bind('<MouseWheel>', self.destroy_entry)
        self.tree.bind("<<TreeviewOpen>>", self.destroy_on_open)
        self.tree.bind("<<TreeviewClose>>", self.destroy_on_close)
        self.tree.bind("<<ScrollEvent>>", self.destroy_entry)
        self.tree.bind("<Leave>", self.tooltip_destroy)
        
        #self.tree.tag_configure('focus', background='yellow')
        self.tree.bind("<Motion>", self.toolTip)
        self.last_focus = None
        

        if self.root:
            # Limit windows minimum dimensions
            self.root.update_idletasks()
            self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())
            self.root.mainloop()
        else:
            pass
    
    def toolTip(self, event):
        
        if event.x > self.text_width:
            rowid = self.tree.identify_row(event.y)
            column = self.tree.identify_column(event.x) 
            
            try:
                x,y,width,height = self.tree.bbox(rowid, column)
            except:
                return
            # not popup if accidentally click any parent key's value field
            is_parent = self.tree.get_children(rowid)
            if is_parent:
                return
            else:
                pass

            x += self.tree.winfo_rootx() + 45
            y += self.tree.winfo_rooty() + 25
            if rowid != self.last_focus:
                if self.last_focus:
                    #self.tree.item(self.last_focus, tags=[])
                    self.tooltip.destroy()
                self.tooltip = tk.Toplevel()
                self.tooltip.wm_overrideredirect(True)
                self.tooltip.wm_geometry("+%d+%d" % (x, y))
                label = tk.Label(self.tooltip, text="Double click to edit, then press Enter to save", justify='left',
                               background="#ffffff", relief='solid', borderwidth=1,
                               wraplength = 130)
                label.pack(ipadx=1)
                #self.tree.item(rowid, tags=['focus'])
                self.last_focus = rowid
        else:
            self.tooltip_destroy()
    
    def tooltip_destroy(self, event=None):  
        try:
            self.tooltip.destroy()
            return
        except:
            return
    
    def destroy_entry(self, event):
        global click_num
        try:
            self.AddButt.destroy()
            self.DelButt.destroy()
        except:
            pass
        try:
            if_entry_ex = self.entryPopup.winfo_exists()
        except:
            if_entry_ex = None
        if if_entry_ex:
            self.entryPopup.destroy()
            click_num -= 1
        else:
            return
    
    def des_wid(self,*args):
        self.tree.yview(*args)
        try:
            self.AddButt.destroy()
            self.DelButt.destroy()
        except:
            pass
        self.entry_del()
        return

    def destroy_on_open(self, event=None):
        global click_num
        if click_num == 1:
            jfields = self.entryPopup.json_current_field
        else:
            return
        # The parent key opened
        par = self.tree.selection()[0]
        name = str(self.tree.item(par)['text'])
        if not name in jfields:
            click_num -= 1
            self.entryPopup.destroy()
        else:
            pass
    def destroy_on_close(self, event):
        global click_num
        # The parent key closed
        par = self.tree.selection()[0]
        if_par = self.tree.get_children(par)
        if if_par and click_num == 1:
            click_num -= 1
            self.entryPopup.destroy()
            return
        else:
            return
            
    
    def onDoubleClick(self, event):
        global click_num
           
        if click_num == 1:
            # close previous popups and save
            self.entryPopup.click_other()
        # No existing entry
        elif click_num == 0:
            pass

        # what row and column was clicked on
        rowid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        
        # get column position info
        try:
            x,y,width,height = self.tree.bbox(rowid, column)
            # Update valid width if dragged
            self.text_width = self.fr_width - width
        except:
            return

        # Get full path of the selscted value
        item_iid = self.tree.selection()[0]
        json_path = str([self.tree.item(item_iid)['text']])
        fullpath_str = self.json_field(item_iid,json_path)
        
        text = self.tree.item(rowid, 'values')
        key = self.tree.item(rowid, 'text')
        # not popup if accidentally click any parent key's value field
        is_parent = self.tree.get_children(self.tree.selection()[0])
        if is_parent:
            return
        else:
            pass
        
        # Only pop up if click on values' side
        if event.x > self.text_width:
            # Allow editting
            if self.edit:
                # place Entry popup properly         
                self.entryPopup = EntryPopup(self.tree, rowid, text, fullpath_str, self.data, self.datapath)
                self.entryPopup.place(x=x+int(width/2.2), y=y+10, anchor=CENTER, width=width, height=15)
                click_num += 1
            else:
                messagebox.showerror("ERROR", "Cannot edit while WormWatcher is running!")
                return
        else:
            return

    # Button show up to add to the selected list
    def onSingleClick_list(self, event):
        global if_lst
        self.tooltip_destroy()
        rowid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        # if any tree item selected
        try:
            x,y,width,height = self.tree.bbox(rowid, column)
            self.text_width = self.fr_width - width
        # Destroy any entry field when single clicking the blank field
        except:
            self.entry_del()
            return
        is_parent = self.tree.get_children(self.tree.selection()[0])
        if is_parent or event.x < self.text_width:
            self.entry_del()
        else:
            pass

        self.item_iid = self.tree.selection()[0]
        self.list_name = str(self.tree.item(self.item_iid)['text'])
        self.jsonfullpath_str = self.json_field(self.item_iid,str([self.list_name]))         

        # Button show up Only for list type key
        if self.item_iid in if_lst:
            try:
                self.AddButt.destroy()
                self.DelButt.destroy()
            except:
                pass
            if event.x < self.text_width:
                self.AddButt = Button(self.tree, text = "Add", command=self.button_add)
                self.AddButt.place(x=x+145, y=y+10, anchor=CENTER, width=50, height=height)
                j_str = self.json_field(self.item_iid, str([str(self.tree.item(self.item_iid)['text'])]))
                exec("global j_lst;j_lst = "+j_str)
                # Empty list
                if not j_lst:
                    self.AddButt["state"] = "disabled"
            return 
        # Delete button for list's elements
        elif self.item_iid in self.lst_elem:
            try:
                self.AddButt.destroy()
                self.DelButt.destroy()
            except:
                pass
            if event.x < self.text_width:
                self.DelButt = Button(self.tree, text = "Delete", command=self.button_del)
                self.DelButt.place(x=x+145, y=y+10, anchor=CENTER, width=50, height=height)
                parent_iid = self.tree.parent(self.item_iid)
                jpar_str = self.json_field(parent_iid, str([str(self.tree.item(parent_iid)['text'])]))
                exec("global j_lst;j_lst = "+jpar_str)
                elem_num = len(j_lst)
                # Just one item in a list, not allow to delete
                if elem_num == 1:
                    self.DelButt["state"] = "disabled"
            return
        else:    
            try:
                self.AddButt.destroy()
                self.DelButt.destroy()
                return
            except:
                return       
    
    def button_add(self):
        global if_lst
        # Get the format of the item to be added
        exec("global j_lst;j_lst = "+self.jsonfullpath_str)
        key_name = len(j_lst)
        dic={}
        dic[key_name] = copy.deepcopy(j_lst[0])
        
        #Calculate the row and column of the added plate
        #   First find out how many columns are in each row
        num_cols = 0
        for plate in j_lst:
            if plate['row'] != 0:
                break
            num_cols += 1

        #   Then just add the next row/col based on the row/col of the last plate in the tree
        last_row = j_lst[-1]['row']
        last_col = j_lst[-1]['col']

        if last_col + 1 >= num_cols:
            new_row = last_row + 1
            new_col = 0
        else:
            new_row = last_row
            new_col = last_col +1

        dic[key_name]['row'] = new_row
        dic[key_name]['col'] = new_col
        dic[key_name]['name'] = f'NewPlate{new_row}_{new_col}'

        # Add to treeview
        j_tree(self.tree, self.item_iid, dic)
        # Save to Json
        j_lst.append(dic[key_name])
        with open(self.datapath, 'w') as f:
            f.write(json.dumps(self.data, indent=4))
        elem = self.tree.get_children(self.item_iid)[-1]
        self.lst_elem.append(elem)
        self.entry_del()
        return

    def button_del(self):
        parent_iid = self.tree.parent(self.item_iid)
        ind = self.tree.item(self.item_iid)["text"]
        jpar_str = self.json_field(parent_iid, str([str(self.tree.item(parent_iid)['text'])]))
        exec("global j_lst;j_lst = "+jpar_str)
        # Whether the deleted element is the last one in the list
        if ind == len(j_lst)-1:
            pass
        else:
            elem = self.tree.get_children(parent_iid)
            for i in elem[ind:]:
                num = int(self.tree.item(i)['text']) - 1
                self.tree.item(i, text=num)
        del j_lst[ind]
        with open(self.datapath, 'w') as f:
            f.write(json.dumps(self.data, indent=4))
        self.tree.delete(self.item_iid)
        self.DelButt.destroy()
        self.lst_elem.remove(self.item_iid)
        self.entry_del()
        return

    def json_field(self, item_iid, json_path):
        '''return the selected item's full json path (a string)
        json_path is the json current field should be in str([])'''
        for i in range(20):
            parent_iid = self.tree.parent(item_iid)
            if parent_iid:
                item_iid = parent_iid
                json_path = str([self.tree.item(item_iid)['text']]) + json_path
            else:
                fullpath_str = "self.data" + json_path
                # Need to convert back to "dict" to access json later
                break
        return fullpath_str
    # Destroy any existing entry field
    def entry_del(self):
        global click_num
        try:
            if_entry_ex = self.entryPopup.winfo_exists()
        except:
            if_entry_ex = None
        if if_entry_ex:
            self.entryPopup.destroy()
            click_num -= 1
            return
        else:
            return

    '''def next_popup(self, event):
        pass'''

        

class EntryPopup(Entry):

    def __init__(self, parent, iid, text, json_path, json_file, datapath, **kw):
       
        super().__init__(parent, justify = "center", **kw)
        self.tv = parent
        self.iid = iid
        self.json_current_field = json_path
        self.data = json_file
        self.datapath = datapath

        self.insert(0, text) 
        # self['state'] = 'readonly'
        # self['readonlybackground'] = 'white'
        # self['selectbackground'] = '#1BA1E2'
        self['exportselection'] = False

        self.focus_force()
        self.bind("<Return>", self.on_return)
        self.bind("<Control-a>", self.select_all)
        self.bind("<Escape>", self.escape)
        
        #self.bind("<Tab>", tree_view.next_popup)
    
    # Save to json when pressing "retrun"
    def on_return(self, event):
        global click_num, path_lst
        
        click_num -= 1
        # If enter file path
        if self.iid in path_lst or "\\" in self.get():
            if not os.path.exists(self.get()):
                # try to create a new path
                try:
                    os.makedirs(self.get())
                    p = pathlib.PureWindowsPath(self.get())
                    pa = p.as_posix()
                    self.tv.item(self.iid, values=pa)
                except:
                    messagebox.showerror("ERROR", "Invalid file path!")
                    self.destroy()
                    return
            else:
                p = pathlib.PureWindowsPath(self.get())
                pa = p.as_posix()
                self.tv.item(self.iid, values=pa)
        else:
            self.tv.item(self.iid, values=self.get())

        # Data type
        if "true" in self.get():
            save_json = self.json_current_field + "=True"
        elif "false" in self.get():
            save_json = self.json_current_field + "=False"
        else:
            try:
                val = float(self.get())
                if "." in self.get():
                    save_json = self.json_current_field + "=float(self.get())"
                else:
                    save_json = self.json_current_field + "=int(self.get())"
            # String
            except ValueError:
                save_json = self.json_current_field + "=self.get()"
        exec(save_json)
        
        with open(self.datapath, 'w') as f:
            f.write(json.dumps(self.data, indent=4))
        self.destroy()
    
    # Save to json when double-clicking another entry field   
    def click_other(self, *ignore):
        global click_num, path_lst
        
        click_num -= 1
        if self.iid in path_lst or "\\" in self.get():
            if not os.path.exists(self.get()):
                messagebox.showerror("ERROR", "Path could not be created!")
                self.destroy()
                return
            else:
                p = pathlib.PureWindowsPath(self.get())
                pa = p.as_posix()
                self.tv.item(self.iid, values=pa)
        else:
            self.tv.item(self.iid, values=self.get())
        # Data type
        if "true" in self.get():
            save_json = self.json_current_field + "=True"
        elif "false" in self.get():
            save_json = self.json_current_field + "=False"
        else:
            try:
                val = float(self.get())
                if "." in self.get():
                    save_json = self.json_current_field + "=float(self.get())"
                else:
                    save_json = self.json_current_field + "=int(self.get())"
            except ValueError:
                save_json = self.json_current_field + "=self.get()"
        exec(save_json)
        with open(self.datapath, 'w') as f:
            f.write(json.dumps(self.data, indent=4))
        self.destroy()

    def select_all(self, *ignore):
        ''' Set selection on the whole text '''
        self.selection_range(0, 'end')

        # returns 'break' to interrupt default key-bindings
        return 'break'

    def escape(self, event):
        global click_num
        click_num -= 1
        self.destroy()

    


if __name__ == "__main__":

    #json_name = r"C:\WormWatcher\config.json"
    
    root = tk.Tk()
    root.geometry("600x680+50+20")
    fr = tk.Frame(root)
    fr.pack(fill="x")

    # call it with
    JsonTreeView(frame = fr)
    
    root.mainloop()
