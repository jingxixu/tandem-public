import collections
import multiprocessing as mp
import os
import queue
import shutil
import threading

import dominate
import imageio
import numpy as np


def get_tableau_palette():
    """Get Tableau color palette (10 colors) https://www.tableau.com/.
    Returns:
        palette: 10x3 uint8 array of color values in range 0-255 (each row is a color)
    """
    palette = np.array([[ 78,121,167], # blue
                        [255, 87, 89], # red
                        [ 89,169, 79], # green
                        [242,142, 43], # orange
                        [237,201, 72], # yellow
                        [176,122,161], # purple
                        [255,157,167], # pink 
                        [118,183,178], # cyan
                        [156,117, 95], # brown
                        [186,176,172]  # gray
                        ],dtype=np.uint8)
    return palette


def mkdir(path, clean=False):
    """Make directory.
    
    Args:
        path: path of the target directory
        clean: If there exist such directory, remove the original one or not
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
        

def imretype(im, dtype):
    """Image retype.
    
    Args:
        im: original image. dtype support: float, float16, float32, float64, uint8, uint16
        dtype: target dtype. dtype support: float, float16, float32, float64, uint8, uint16
    
    Returns:
        image of new dtype
    """
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))

    assert np.min(im) >= 0 and np.max(im) <= 1

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im


def imwrite(path, obj):
    """Save Image.
    
    Args:
        path: path to save the image. Suffix support: png or jpg or gif
        image: array or list of array(list of image --> save as gif). Shape support: WxHx3 or WxHx1 or 3xWxH or 1xWxH
    """
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()


def multithreading_exec(num, q, fun, blocking=True):
    """Multi-threading Execution.
    
    Args:
        num: number of threadings
        q: queue of args
        fun: function to be executed
        blocking: blocking or not (default True)
    """
    class Worker(threading.Thread):
        def __init__(self, q, fun):
            super().__init__()
            self.q = q
            self.fun = fun
            self.start()

        def run(self):
            while True:
                try:
                    args = self.q.get(block=False)
                    self.fun(*args)
                    self.q.task_done()
                except queue.Empty:
                    break
    thread_list = [Worker(q, fun) for i in range(num)]
    if blocking:
        for t in thread_list:
            if t.is_alive():
                t.join()


def html_visualize(web_path, data, ids, cols, others=[], title='visualization', threading_num=10):
    """Visualization in html.
    
    Args:
        web_path: string; directory to save webpage. It will clear the old data!
        data: dict; 
            key: {id}_{col}. 
            value: figure or text
                - figure: ndarray --> .png or [ndarrays] --> .gif
                - text: string or [string]
        ids: [string]; name of each row
        cols: [string]; name of each column
        others: (optional) [dict]; other figures
            - name: string; name of the data, visualize using h2()
            - data: string or ndarray(image)
            - height: (optional) int; height of the image (default 256)
        title: (optional) string; title of the webpage (default 'visualization')
        threading_num: (optional) int; number of threadings for imwrite (default 10)
    """
    figure_path = os.path.join(web_path, 'figures')
    mkdir(web_path, clean=True)
    mkdir(figure_path, clean=True)
    q = queue.Queue()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            q.put((os.path.join(figure_path, key + '.png'), value))
        if isinstance(value, list) and isinstance(value[0], np.ndarray):
            q.put((os.path.join(figure_path, key + '.gif'), value))
    multithreading_exec(threading_num, q, imwrite)

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        with dominate.tags.table(border=1, style='table-layout: fixed;'):
            with dominate.tags.tr():
                with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                    dominate.tags.p('id')
                for col in cols:
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', ):
                        dominate.tags.p(col)
            for id in ids:
                with dominate.tags.tr():
                    bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                        for part in id.split('_'):
                            dominate.tags.p(part)
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                            value = data[f'{id}_{col}']
                            if isinstance(value, str):
                                dominate.tags.p(value)
                            elif isinstance(value, list) and isinstance(value[0], str):
                                for v in value:
                                    dominate.tags.p(v)
                            elif isinstance(value, list):
                                dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.gif'.format(id, col)))
                            else:
                                dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.png'.format(id, col)))
        for idx, other in enumerate(others):
            dominate.tags.h2(other['name'])
            if isinstance(other['data'], str):
                dominate.tags.p(other['data'])
            else:
                imwrite(os.path.join(figure_path, '_{}_{}.png'.format(idx, other['name'])), other['data'])
                dominate.tags.img(style='height:{}px'.format(other.get('height', 256)),
                    src=os.path.join('figures', '_{}_{}.png'.format(idx, other['name'])))
    with open(os.path.join(web_path, 'index.html'), 'w') as fp:
        fp.write(web.render())