#%% Intro
#? Testing FastAPI

#%% Imports
from fastapi import FastAPI, HTTPException

#%% [Hello World] [Todolist app]
app = FastAPI()
items = []

@app.get("/")
def root():
    return {"Hello": "World"}

#* New end point. To Post Item
@app.post("/items")
def create_item(item: str):
    items.append(item)
    return items

#* New end point. To view the Item from Items
@app.get("/items/{item_id}")
def get_item(item_id: int) -> str:
    if item_id < len(items):
        item = items[item_id]
        return item
    else:
        raise HTTPException(status_code=404, detail="Item Not Found")
    

#?_______________[Comment]_________________
#? Test out the items addition with following cmd:
#! curl -X POST -H "Content-Type: application/json" 'http://127.0.0.1:8000/items?item=apple'
#? Get item using item index (at the end):
#! curl -X GET http://127.0.0.1:8000/items/1
#?_________________________________________

# %%
