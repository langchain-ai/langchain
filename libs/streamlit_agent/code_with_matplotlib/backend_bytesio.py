import base64
from io import BytesIO

import matplotlib
import sys

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureManagerBase, ShowBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PY3 = sys.version_info[0] >= 3

images = []

rcParams = matplotlib.rcParams


class Show(ShowBase):
    def __call__(self, **kwargs):
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return

        for manager in managers:
            manager.show(**kwargs)

    def mainloop(self):
        pass


show = Show()


# from pyplot API
def draw_if_interactive():
    if matplotlib.is_interactive():
        figManager = Gcf.get_active()
        if figManager is not None:
            figManager.canvas.show()


# from pyplot API
def new_figure_manager(num, *args, **kwargs):
    FigureClass = kwargs.pop("FigureClass", Figure)
    figure = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, figure)


# from pyplot API
def new_figure_manager_given_figure(num, figure):
    canvas = FigureCanvasInterAgg(figure)
    manager = FigureManagerInterAgg(canvas, num)
    return manager


# from pyplot API
class FigureCanvasInterAgg(FigureCanvasAgg):
    def __init__(self, figure):
        FigureCanvasAgg.__init__(self, figure)

    def show(self):
        FigureCanvasAgg.draw(self)

        b = BytesIO()
        self.print_png(b)
        global images
        images.append(b)

    def draw(self):
        FigureCanvasAgg.draw(self)
        self.show()


class FigureManagerInterAgg(FigureManagerBase):
    def __init__(self, canvas, num):
        FigureManagerBase.__init__(self, canvas, num)
        self.canvas = canvas
        self._num = num
        self._shown = False

    def show(self, **kwargs):
        self.canvas.show()
        Gcf.destroy(self._num)


class DisplayDataObject:
    def __init__(self, plot_index, width, image_bytes):
        self.plot_index = plot_index
        self.image_width = width
        self.image_bytes = image_bytes

    def _repr_display_(self):
        image_bytes_base64 = base64.b64encode(self.image_bytes)
        if PY3:
            image_bytes_base64 = image_bytes_base64.decode()
        body = {
            "plot_index": self.plot_index,
            "image_width": self.image_width,
            "image_base64": image_bytes_base64,
        }
        return ("pycharm-plot-image", body)


FigureCanvas = FigureCanvasAgg
FigureManager = FigureManagerInterAgg
