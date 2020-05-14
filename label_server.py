import web
#from web.wsgiserver import CherryPyWSGIServer
import os
import pdb
urls = ('/auto_annotate','Upload')
import os.path as osp
import string
import _thread
import logging
import urllib
import base64
from queue import Queue
import time
import random
import json

import predict_with_print_box as yolo_demo

taskQueue = Queue()
taskInImages = {}
base_path = "/nfs/"

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='labellog.txt',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    #'%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    '%(asctime)s - %(message)s'
                    #日志格式
                    )
logging.info('------------------------------------label server start-----------------------------')

def get_code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))
def get_32code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 32))


class Upload:
    def GET(self):
        x = web.input()
        print(x)
        #web.header('content-type', 'text/json')
        web.header("Access-Control-Allow-Origin", "*")
        web.header("Access-Control-Allow-Credentials", "true")
        web.header('Access-Control-Allow-Headers',  'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
        web.header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')
        return """<html><head></head><body>please send data in post
</body></html>"""
 
    def POST(self):
        try:
            web.header("Access-Control-Allow-Origin", "*")
            web.header("Access-Control-Allow-Credentials", "true")
            web.header('Access-Control-Allow-Headers',  'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
            web.header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')            
#            pdb.set_trace()
#            type_ = 1
#            x = web.input(_method = 'POST')
            x = web.data()
#            logging.info(x)
#            print("x:", x)
#            for i, key in enumerate(x):
#                if key == "type":
#                    type_ = x[key]
#                else:
#                    y = key  
#            y = json.loads(y)
            x = json.loads(x.decode())
            type_ = x['annotateType']
            taskId = get_code() 
            taskInImages = {}
            taskInImages[taskId] = {"input":{'type':type_, 'data':x},"output":{"annotations":[]}}
#            print("type:", type_)
            print("Random_code:", taskId)
            logging.info(taskId)
#            pdb.set_trace()
            web.t_queue.put(taskInImages)
#            pdb.set_trace()
#            print("web:", web.t_queue.queue)
            return {"code":200, "msg":"", "data":taskId}         
        except Exception as e:
                logging.error("Error post")
                logging.error(e)            
                print(e)
                print("Error Post")
                return 'post error'
def bgProcess():
    global taskQueue
    while True:
        try: 
            task_dict =  taskQueue.get()  
            for taskId in task_dict:
#                pdb.set_trace()
                id_list = []
                image_path_list = []
                type_ = task_dict[taskId]["input"]['type']
                for file in task_dict[taskId]["input"]['data']["files"]:
#                    pdb.set_trace()
                    id_list.append(file["id"])
                    image_path_list.append(base_path+file["url"])
                label_list = task_dict[taskId]["input"]['data']["labels"]
                image_num = len(image_path_list)
                if image_num < 16:
                    for i in range(16-image_num):
                        image_path_list.append(image_path_list[0])
                        id_list.append(id_list[0])
                logging.info(image_num)
                logging.info(image_path_list)
#                print("image_num", image_num)
                print("image_path_list", image_path_list)                
#                print("id_list:", id_list)
#                print("image_path_list:", image_path_list)
#                print("label_list:", label_list)
                annotations = yolo_obj.yolo_inference(type_, id_list, image_path_list, label_list)
                annotations = annotations[0:image_num]
#                pdb.set_trace()
                result = {"annotations":annotations}
                print("result", result)
                send_data = json.dumps(result).encode()
#                logging.info(send_data)
                url = 'http://10.5.18.239:8200/api/data/datasets/files/annotations/auto/'+taskId
                headers = {'Content-Type':'application/json'}   
                req = urllib.request.Request(url, headers=headers)
                response = urllib.request.urlopen(req, data=send_data, timeout=5)              
#                pdb.set_trace()
#                print(response.read())
                logging.info(response.read())
                logging.info("End mayechi")
                print("End mayechi")
        except Exception as e:
            logging.error("Error bgProcess")
            logging.error(e)
            print("Error bgProcess")
            print(e)
#        pass
#        if len(taskInImages) > 500:
#            print('------------clear taskInImages----------------------')
#            taskInImages.clear()
        time.sleep(0.01)
            
 
def bg_thread(no, interval):
    print('bg_thread on')
    bgProcess()

if __name__ == "__main__":  
    yolo_obj = yolo_demo.YoloInference()
   #os.system('taskkill /f /im %s' % '\"Topaz A.I. Gigapixel.exe\"')
    _thread.start_new_thread(bg_thread, (5,5))
   #CherryPyWSGIServer.ssl_certificate = "./certificate.crt"
   #CherryPyWSGIServer.ssl_private_key = "./privateKey.key"
    app = web.application(urls, globals())
    
    web.t_queue = taskQueue
    web.taskInImages = taskInImages
    app.run()
