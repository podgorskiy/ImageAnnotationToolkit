import anntoolkit
import imageio
import os
import pickle
import numpy as np


LIBRARY_PATH = 'library'
SAVE_PATH = 'art_face_save.pth'
# LIBRARY_PATH = 'images'
# SAVE_PATH = 'save_cio.pth'


FILTER = 'Fiship'


class App(anntoolkit.App):
    def __init__(self):
        super(App, self).__init__(title='Test')

        self.path = LIBRARY_PATH
        self.paths = []
        for dirName, subdirList, fileList in os.walk(self.path):
            self.paths += [os.path.relpath(os.path.join(dirName, x), self.path) for x in fileList if x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')]
        self.paths.sort()
        self.iter = -1
        self.annotation = {}
        if os.path.exists(SAVE_PATH):
            with open(SAVE_PATH, 'rb') as f:
                self.annotation = pickle.load(f)
        if '__banned' not in self.annotation.keys():
            self.annotation['__banned'] = set()

        ################ to delete
        # with open('save_cio_2.pth', 'rb') as f:
        #     annotation2 = pickle.load(f)
        # print(annotation2)
        # for k, x in annotation2.items():
        #     self.annotation['rev/' + k] = x
        #     if not os.path.exists(os.path.join(self.path, 'rev/' + k)):
        #         print('Not found: %s' % 'rev/' + k)
        #
        # with open(SAVE_PATH, 'wb') as f:
        #     pickle.dump(self.annotation, f)
        ################

        to_delete = []
        for p, x in self.annotation.items():
            if len(x) == 0 and p != '__banned':
                to_delete.append(p)
        for p in to_delete:
            del self.annotation[p]

        for p in self.paths:
            if p in self.annotation['__banned'] and p in self.annotation:
                self.annotation['__banned'].remove(p)

        self.moving = None
        self.load_next()

    def load_next(self):
        while True:
            self.iter += 1
            self.iter = self.iter % len(self.paths)
            try:
                if FILTER in self.paths[self.iter] and self.paths[self.iter] not in self.annotation['__banned']:
                    im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
                    self.set_image(im)
                    break
            except ValueError:
                continue

    def load_next_not_annotated(self):
        while True:
            self.iter += 1
            self.iter = self.iter % len(self.paths)
            if FILTER in self.paths[self.iter] and self.paths[self.iter] not in self.annotation['__banned']:
                k = self.paths[self.iter]
                if k not in self.annotation:
                    break
                if self.iter == 0:
                    break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
        except ValueError:
            self.load_next_not_annotated()

    def load_prev(self):
        while True:
            self.iter -= 1
            self.iter = (self.iter + len(self.paths)) % len(self.paths)
            if FILTER in self.paths[self.iter] and self.paths[self.iter] not in self.annotation['__banned']:
                try:
                    im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
                    self.set_image(im)
                    break
                except ValueError:
                    continue

    def load_prev_not_annotated(self):
        while True:
            self.iter -= 1
            self.iter = (self.iter + len(self.paths)) % len(self.paths)
            if FILTER in self.paths[self.iter] and self.paths[self.iter] not in self.annotation['__banned']:
                k = self.paths[self.iter]
                if k not in self.annotation:
                    break
                if self.iter == 0:
                    break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
        except ValueError:
            self.load_prev_not_annotated()

    def on_update(self):
        k = self.paths[self.iter]
        self.text(k, 10, 10)
        if k in self.annotation:
            self.text("Points count %d" % len(self.annotation[k]), 10, 50)
            for i, p in enumerate(self.annotation[k]):
                if i == self.moving:
                    self.point(*p, (0, 255, 0, 250))
                else:
                    self.point(*p, (255, 0, 0, 250))

    def on_mouse_button(self, down, x, y, lx, ly):
        k = self.paths[self.iter]
        if down:
            if k in self.annotation:
                points = np.asarray(self.annotation[k])
                if len(points) > 0:
                    point = np.asarray([[lx, ly]])
                    d = points - point
                    d = np.linalg.norm(d, axis=1)
                    i = np.argmin(d)
                    if d[i] < 2:
                        self.moving = i
                        print(i, d[i])

        if not down:
            if k not in self.annotation:
                self.annotation[k] = []
            if self.moving is not None:
                self.annotation[k][self.moving] = (lx, ly)
                self.moving = None
            else:
                self.annotation[k].append((lx, ly))
            with open(SAVE_PATH, 'wb') as f:
                pickle.dump(self.annotation, f)

    def on_mouse_position(self, x, y, lx, ly):
        if self.moving is not None:
            k = self.paths[self.iter]
            self.annotation[k][self.moving] = (lx, ly)

    def on_keyboard(self, key, down, mods):
        if down:
            if key == anntoolkit.KeyLeft:
                self.load_prev()
            if key == anntoolkit.KeyRight:
                self.load_next()
            if key == anntoolkit.KeyUp:
                self.load_next_not_annotated()
            if key == anntoolkit.KeyDown:
                self.load_prev_not_annotated()
            if key == anntoolkit.KeyDelete:
                k = self.paths[self.iter]
                if k in self.annotation:
                    del self.annotation[k]
                with open(SAVE_PATH, 'wb') as f:
                    pickle.dump(self.annotation, f)
            if key == anntoolkit.KeyBackspace:
                k = self.paths[self.iter]
                if k in self.annotation and len(self.annotation[k]) > 0:
                    self.annotation[k] = self.annotation[k][:-1]
                with open(SAVE_PATH, 'wb') as f:
                    pickle.dump(self.annotation, f)

            if key == 'B':
                self.annotation['__banned'].add(self.paths[self.iter])
                # self.load_next()

app = App()

app.run()
