from ipywidgets import widgets
from IPython.core.magics.namespace import NamespaceMagics
from IPython import get_ipython
import numpy as np
from types import ModuleType, FunctionType

__all__ = ['workspace']


class workspace(object):

    """Print the workspace of a Ipython notebook. Import the workspace class
    and run workspace()
    """

    def __init__(self, ipython):
        # ipython = get_ipython()
        # ipython.user_ns_hidden['widgets'] = widgets
        # ipython.user_ns_hidden['NamespaceMagics'] = NamespaceMagics
        self._closed = False
        self.namespace = NamespaceMagics()
        self.namespace.shell = ipython.kernel.shell
        self._typeList = [bool, (int, float), np.ndarray, list, dict, bool,
                          str, ModuleType, FunctionType]
        self._typeListName = ['All', 'Int/float', 'Matrix', 'List', 'Dict',
                              'Bool', 'String', 'Module', 'Function']
        self._tab, self._tablab = zip(
            *[self._createTab(widgets.VBox(), []) for k in self._typeListName])

        self._popout = widgets.Tab()
        self._popout.description = "Workspace"
        self._popout.button_text = self._popout.description
        self._popout.children = list(self._tab)
        [self._popout.set_title(page, name)
         for page, name in enumerate(self._typeListName)]

        self._ipython = ipython
        self._ipython.events.register('post_run_cell', self._fill)

    def close(self):
        """Close and remove hooks."""
        if not self._closed:
            self._ipython.events.unregister('post_run_cell', self._fill)
            self._popout.close()
            self._closed = True

    def _fill(self):
        """Fill self with variable information."""
        values = self.namespace.who_ls()
        # Scan shape & type :
        valShape, valType = [], []
        for k, v in enumerate(values):
            # Get type :
            valType.append(type(eval(v)))
            # Get shape :
            try:
                try:
                    valShape.append(str(eval(v).shape))
                except:
                    valShape.append(str(len(eval(v))))
            except:
                valShape.append('')
        # Fill each tab :
        for num, tab in enumerate(self._tablab):
            tab.value = '<table class="table table-bordered table-striped"><tr><th>Name</th><th>Type</th><th>Value</th><th>Shape</th</tr><tr><td>' + \
            '</td></tr><tr><td>'.join(['{0}</td><td>{1}</td><td>{2}<td>{3}</td>'.format(v, type(eval(v)).__name__, str(eval(v)), valShape[k]) for k, v in enumerate(values) if isinstance(eval(v), self._typeList[num]) or num == 0]) + \
            '</td></tr></table>'

    def _ipython_display_(self):
        """Called when display() or pyout is used to display the variable
        Inspector.
        """
        self._popout._ipython_display_()

    @staticmethod
    def _createTab(var, label):
        """Create each tab of the table"""
        var = widgets.VBox()
        var.overflow_y = 'scroll'
        var.overflow_x = 'scroll'
        label = widgets.HTML(value='Not hooked')
        var.children = [label]
        return var, label

if __name__ == '__main__':
    wksp = workspace(get_ipython())
