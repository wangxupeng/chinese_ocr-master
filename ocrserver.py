import time
import tornado.ioloop
import tornado.web
import os
import ocr
import shutil
from PIL import Image
from glob import glob
import numpy as np
from ctpn.text_detect import load_tf_model
image_files = glob('./test_images/*.*')

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        start = time.time()
        result_dir = './test_result'
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.mkdir(result_dir)

        for image_file in sorted(image_files):
            image = np.array(Image.open(image_file).convert('RGB'))
            t = time.time()
            result, image_framed = ocr.model(image)
            output_file = os.path.join(result_dir, image_file.split('/')[-1])
            Image.fromarray(image_framed).save(output_file)
            print("Mission complete, it took {:.3f}s".format(time.time() - t))
            print("单条预测时间", time.time() - t)
            print("\nRecognition Result:\n")
            for key in result:
                print(result[key][1])
        print("总计预测时间", time.time() - start)
        self.write("1233")
class OCRHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        ret = {'result': 'OK'}
        result_dir = './test_result'
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.mkdir(result_dir)
        upload_path = os.path.join(os.path.dirname(__file__), 'demo')  # 文件的暂存路径
        file_metas = self.request.files.get('file', None)  # 提取表单中‘name’为‘file’的文件元数据
        filename = file_metas[0]['filename']
        file_path = os.path.join(upload_path, filename)
        with open(file_path, 'wb') as up:
            up.write(file_metas[0]['body'])
        image = np.array(Image.open(file_path).convert('RGB'))
        t = time.time()
        result, image_framed = ocr.model(image)
        output_file = os.path.join(result_dir, file_path.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("单条预测时间", time.time() - t)
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
        self.write("13213")

def make_app():
    return tornado.web.Application(
        [
            (r"/upload", OCRHandler),
        ]
    )




if __name__ == "__main__":
    # sess, net = load_tf_model()
    app = make_app()
    app.listen(8888)
    print('Handler Init Success')
    tornado.ioloop.IOLoop.current().start()
