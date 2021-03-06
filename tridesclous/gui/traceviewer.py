import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pandas as pd

from .base import WidgetBase
from .tools import TimeSeeker
from ..tools import median_mad
from ..dataio import _signal_types



class MyViewBox(pg.ViewBox):
    doubleclicked = QtCore.pyqtSignal()
    gain_zoom = QtCore.pyqtSignal(float)
    xsize_zoom = QtCore.pyqtSignal(float)
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #~ self.disableAutoRange()
    def mouseClickEvent(self, ev):
        ev.accept()
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.ignore()
    def wheelEvent(self, ev):
        if ev.modifiers() == QtCore.Qt.ControlModifier:
            z = 10 if ev.delta()>0 else 1/10.
        else:
            z = 1.3 if ev.delta()>0 else 1/1.3
        self.gain_zoom.emit(z)
        ev.accept()
    def mouseDragEvent(self, ev):
        ev.accept()
        self.xsize_zoom.emit((ev.pos()-ev.lastPos()).x())


class BaseTraceViewer(WidgetBase):
    
    def __init__(self, spikesorter = None, shared_view_with = [], 
                    mode = 'memory', signal_type = 'filtered', parent=None):
        WidgetBase.__init__(self, parent)
    
        self.spikesorter = spikesorter
        self.dataio = self.spikesorter.dataio
        self.mode = mode
        self.signal_type = signal_type
        
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        # Can share view with other trace viewer : simultanetly view filter + full band
        self.shared_view_with = shared_view_with

        self.create_toolbar()
        self.layout.addWidget(self.toolbar)
        
        # create graphic view and plot item
        self.graphicsview = pg.GraphicsView()
        self.layout.addWidget(self.graphicsview)
        self.initialize_plot()
            
        
        #handle time by segments
        t_starts = self.dataio.segments_range.loc[:, ('filtered','t_start')].copy()
        self.time_by_seg = pd.Series(t_starts, name = 'time', index = self.dataio.segments_range.index)
        
        _params = [{'name': 'auto_zoom_on_select', 'type': 'bool', 'value': True },
                           {'name': 'zoom_size', 'type': 'float', 'value':  0.08, 'step' : 0.001 },
                          {'name': 'plot_threshold', 'type': 'bool', 'value':  True },
                          ]
        self.params = pg.parametertree.Parameter.create( name='Global options', type='group', children = _params)
        self.params.param('plot_threshold').sigValueChanged.connect(self.refresh)
        self.tree_params = pg.parametertree.ParameterTree(parent  = self)
        self.tree_params.header().hide()
        self.tree_params.setParameters(self.params, showTop=True)
        self.tree_params.setWindowTitle(u'Options for signal viewer')
        self.tree_params.setWindowFlags(QtCore.Qt.Window)
        
        self.change_segment(0)
        self.refresh()
    
    def create_toolbar(self):
        tb = self.toolbar = QtGui.QToolBar()
        
        #Segment selection
        self.combo_seg = QtGui.QComboBox()
        tb.addWidget(self.combo_seg)
        self.combo_seg.addItems([ 'Segment {}'.format(seg_num) for seg_num in self.dataio.segments_range.index ])
        self._seg_pos = 0
        self.seg_num = self.dataio.segments_range.index[self._seg_pos]
        self.combo_seg.currentIndexChanged.connect(self.on_combo_seg_changed)
        tb.addSeparator()
        
        self.combo_type = QtGui.QComboBox()
        tb.addWidget(self.combo_type)
        self.combo_type.addItems([ signal_type for signal_type in _signal_types ])
        self.combo_type.currentIndexChanged.connect(self.on_combo_type_changed)

        # time slider
        self.timeseeker = TimeSeeker()
        tb.addWidget(self.timeseeker)
        self.timeseeker.time_changed.connect(self.seek)
        
        # winsize
        self.xsize = .5
        tb.addWidget(QtGui.QLabel(u'X size (s)'))
        self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 10.], suffix = 's', siPrefix = True, step = 0.1, dec = True)
        #~ self.spinbox_xsize = pg.SpinBox(value = self.xsize, bounds = [0.001, 10.]) # step = 0.1, dec = True)
        self.spinbox_xsize.sigValueChanged.connect(self.xsize_changed)
        tb.addWidget(self.spinbox_xsize)
        tb.addSeparator()
        self.spinbox_xsize.sigValueChanged.connect(self.refresh)
        
        #
        but = QtGui.QPushButton('auto scale')
        but.clicked.connect(self.auto_scale)
        tb.addWidget(but)
        but = QtGui.QPushButton('settings')
        but.clicked.connect(self.open_settings)
        tb.addWidget(but)
        self.select_button = QtGui.QPushButton('select', checkable = True)
        tb.addWidget(self.select_button)
        
        self._create_toolbar()
    

    def initialize_plot(self):
        self.viewBox = MyViewBox()
        self.plot = pg.PlotItem(viewBox=self.viewBox)
        self.graphicsview.setCentralItem(self.plot)
        self.plot.hideButtons()
        self.plot.showAxis('left', False)
        
        self.viewBox.gain_zoom.connect(self.gain_zoom)
        self.viewBox.xsize_zoom.connect(self.xsize_zoom)
        
        
        self.curves = []
        self.channel_labels = []
        self.threshold_lines =[]
        self.scatters = []
        for c in range(self.dataio.nb_channel):
            color = '#7FFF00'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves.append(curve)
            label = pg.TextItem(str(self.dataio.info['channels'][c]), color=color, anchor=(0, 0.5), border=None, fill=pg.mkColor((128,128,128, 180)))
            self.plot.addItem(label)
            self.channel_labels.append(label)
            
            tc = pg.InfiniteLine(angle = 0., movable = False, pen = pg.mkPen('w'))
            tc.setPos(0.)
            self.threshold_lines.append(tc)
            self.plot.addItem(tc)
            tc.hide()
            
            self.scatters.append({})
        
        self._initialize_plot()
        
        self.gains = None
        self.offsets = None

    def open_settings(self):
        if not self.tree_params.isVisible():
            self.tree_params.show()
        else:
            self.tree_params.hide()
        

    def prev_segment(self):
        self.change_segment(self._seg_pos - 1)
        
    def next_segment(self):
        self.change_segment(self._seg_pos + 1)

    def change_segment(self, seg_pos):
        self._seg_pos  =  seg_pos
        if self._seg_pos<0:
            self._seg_pos = self.dataio.segments_range.shape[0]-1
        if self._seg_pos == self.dataio.segments_range.shape[0]:
            self._seg_pos = 0
        self.seg_num = self.dataio.segments_range.index[self._seg_pos]
        self.combo_seg.setCurrentIndex(self._seg_pos)
        
        lims = self.dataio.segments_range.xs(self.signal_type, axis=1).loc[self.seg_num]
        
        if self.mode == 'memory':
            self.sigs = self.dataio.get_signals(seg_num = self.seg_num, t_start = lims['t_start'], 
                            t_stop = lims['t_stop'], signal_type = self.signal_type)
        elif self.mode == 'file':
            self.sigs = None
        
        self.load_peak_or_spiketrain()

        self.timeseeker.set_start_stop(lims['t_start'], lims['t_stop'], seek = False)
        #~ self.timeseeker.seek(self.time_by_seg[self.seg_num], emit = False)

        if self.isVisible():
            self.refresh()

    
    def on_combo_seg_changed(self):
        s =  self.combo_seg.currentIndex()
        for otherviewer in self.shared_view_with:
            otherviewer.combo.setCurrentIndex(s)
        self.change_segment(s)
    
    def on_combo_type_changed(self):
        s =  self.combo_type.currentIndex()
        self.signal_type = _signal_types[s]
        self.change_segment(self._seg_pos)
        
    
    def xsize_changed(self):
        self.xsize = self.spinbox_xsize.value()
        for otherviewer in self.shared_view_with:
            otherviewer.spinbox_xsize.setValue(self.xsize)
        if self.isVisible():
            self.refresh()
    
    def refresh(self):
        self.seek(self.time_by_seg[self.seg_num], cascade = False)

    def xsize_zoom(self, xmove):
        factor = xmove/100.
        newsize = self.xsize*(factor+1.)
        limits = self.spinbox_xsize.opts['bounds']
        if newsize>0. and newsize<limits[1]:
            self.spinbox_xsize.setValue(newsize)
    
    def auto_scale(self):
        self.estimate_auto_scale()
        self.refresh()
    
    def estimate_auto_scale(self):
        if self.mode == 'memory':
            self.med, self.mad = median_mad(self.sigs, axis = 0)
        elif self.mode == 'file':
            lims = self.dataio.segments.loc[self.seg_num]
            chunk = self.dataio.get_signals(seg_num = self.seg_num, t_start = lims['t_start'], t_stop = lims['t_start']+60., signal_type = self.signal_type)
            self.med, self.mad = median_mad(chunk, axis = 0)
        
        self.med, self.mad = self.med.values, self.mad.values
        self.factor = 1.
        self.gain_zoom(15.)
    
    def gain_zoom(self, factor_ratio):
        self.factor *= factor_ratio
        n = self.dataio.nb_channel
        self.gains = np.ones(n, dtype=float) * 1./(self.factor*max(self.mad))
        self.offsets = np.arange(n)[::-1] - self.med*self.gains
        self.refresh()

    def seek(self, t, cascade=True):
        if cascade:
            for otherviewer in self.shared_view_with:
                otherviewer.seek(t, cascade = False)
        else:
            self.timeseeker.seek(t, emit = False)
            
        self.time_by_seg[self.seg_num] = t
        t1,t2 = t-self.xsize/3. , t+self.xsize*2/3.
        t_start = self.dataio.segments_range.loc[self.seg_num, (self.signal_type,'t_start')]
        sr = self.dataio.sampling_rate
        ind1 = max(0, int((t1-t_start)*sr))
        ind2 = int((t2-t_start)*sr)
        
        if self.gains is None:
            self.estimate_auto_scale()

        #signal chunk
        if self.mode == 'memory':
            #~ chunk = self.sigs.loc[t1:t2]
            chunk = self.sigs.iloc[ind1:ind2]
        elif self.mode == 'file':
            chunk = self.dataio.get_signals(seg_num = self.seg_num, t_start = t1, t_stop = t2, signal_type = self.signal_type)
        
        for c in range(self.dataio.nb_channel):
            self.curves[c].setData(chunk.index.values, chunk.iloc[:, c].values*self.gains[c]+self.offsets[c])
            self.channel_labels[c].setPos(t1, self.dataio.nb_channel-c-1)
        
        inwindow_times, inwindow_label, inwindow_selected = self.get_peak_or_spiketrain_in_window(t1, t2)
        
        if inwindow_times is not None:
            
            for c in range(self.dataio.nb_channel):
                #reset scatters
                for k in self.spikesorter.cluster_labels:
                    if not self.spikesorter.cluster_visible[k]:
                        self.scatters[c][k].setData([], [])
                
                for k in list(self.scatters[c].keys()):
                    if not k in self.spikesorter.cluster_labels:
                        scatter = self.scatters[c].pop(k)
                        self.plot.removeItem(scatter)
                
                # plotted selected
                if 'sel' not in self.scatters[c]:
                    brush = QtGui.QColor( 'magenta')
                    brush.setAlpha(180)
                    pen = QtGui.QColor( 'yellow')
                    self.scatters[c]['sel'] = pg.ScatterPlotItem(pen=pen, brush=brush, size=20, pxMode = True)
                    self.plot.addItem(self.scatters[c]['sel'])
                
                p = chunk.loc[inwindow_times[inwindow_selected]]
                self.scatters[c]['sel'].setData(p.index.values, p.iloc[:,c].values*self.gains[c]+self.offsets[c])
            
            for k in self.spikesorter.cluster_labels:
                if not self.spikesorter.cluster_visible[k]:
                    continue
                p = chunk.loc[inwindow_times[inwindow_label==k]]
                for c in range(self.dataio.nb_channel):
                    color = self.spikesorter.qcolors.get(k, QtGui.QColor( 'white'))
                    if k not in self.scatters[c]:
                        self.scatters[c][k] = pg.ScatterPlotItem(pen=None, brush=color, size=10, pxMode = True)
                        self.plot.addItem(self.scatters[c][k])
                        self.scatters[c][k].sigClicked.connect(self.item_clicked)
                    
                    if self.spikesorter.catalogue[k]['channel_peak_max'] == c:
                        self.scatters[c][k].setBrush(color)
                        self.scatters[c][k].setData(p.index.values, p.iloc[:,c].values*self.gains[c]+self.offsets[c])
                    else:
                        self.scatters[c][k].setData([], [])
            
            self._plot_prediction(t1, t2, chunk, ind1)
        
        n = self.dataio.nb_channel
        for c in range(n):
            if self.params['plot_threshold'] and self.spikesorter.threshold is not None:
                self.threshold_lines[c].setPos(n-c-1 + self.gains[c]*self.mad[c]*self.spikesorter.threshold)
                self.threshold_lines[c].show()
            else:
                self.threshold_lines[c].hide()
        
        self.plot.setXRange( t1, t2, padding = 0.0)
        self.plot.setYRange(-.5, self.dataio.nb_channel-.5, padding = 0.0)



class TraceViewer(BaseTraceViewer):
    def _create_toolbar(self):
        pass
        
    def _initialize_plot(self):
        pass
    
    def load_peak_or_spiketrain(self):
        if self.spikesorter.peak_labels is not None:
            self.seg_peak_labels = self.spikesorter.peak_labels.xs(self.seg_num)
        else:
            self.seg_peak_labels = None

    def get_peak_or_spiketrain_in_window(self, t1, t2):
        if self.seg_peak_labels is None:
            return None, None
        inwindow = self.seg_peak_labels.loc[t1:t2]
        inwindow_label = inwindow.values
        inwindow_times = inwindow.index.values
        
        seg_selection = self.spikesorter.peak_selection.xs(self.seg_num)
        inwindow_selected = seg_selection.loc[inwindow.index].values
        
        return inwindow_times, inwindow_label, inwindow_selected
    
    def _plot_prediction(self,t1, t2, chunk, ind1):
        pass
    
    def on_peak_selection_changed(self):
        selected_peaks = self.spikesorter.peak_selection[self.spikesorter.peak_selection]
        if self.params['auto_zoom_on_select'] and selected_peaks.shape[0]==1:
            seg_num, time= selected_peaks.index[0]
            if seg_num != self.seg_num:
                seg_pos = self.dataio.segments_range.index.tolist().index(seg_num)
                self.combo_seg.setCurrentIndex(seg_pos)
            self.spinbox_xsize.setValue(self.params['zoom_size'])
            self.seek(time)
        else:
            self.refresh()
    
    def item_clicked(self, plot, points):
        if self.select_button.isChecked()and len(points)==1:
            x = points[0].pos().x()
            self.spikesorter.peak_selection[:] = False
            self.spikesorter.peak_selection.loc[(self.seg_num, x)] = True
            
            self.peak_selection_changed.emit()
            self.refresh()
    


class PeelerTraceViewer(BaseTraceViewer):
    def _create_toolbar(self):
        self.plot_buttons = {}
        for name in ['signals', 'prediction', 'residual']:
            self.plot_buttons[name] = but = QtGui.QPushButton(name,  checkable = True)
            but.clicked.connect(self.refresh)
            self.toolbar.addWidget(but)
            
            if name in ['signals', 'prediction']:
                but.setChecked(True)
    
    def _initialize_plot(self):
        self.curves_prediction = []
        self.curves_residuals = []
        for c in range(self.dataio.nb_channel):
            color = '#FF00FF'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves_prediction.append(curve)

            color = '#FFFF00'  # TODO
            curve = pg.PlotCurveItem(pen=color)
            self.plot.addItem(curve)
            self.curves_residuals.append(curve)

   
    def load_peak_or_spiketrain(self):
        self.spiketrains = self.dataio.get_spiketrains(self.seg_num)

    def get_peak_or_spiketrain_in_window(self, t1, t2):
        if self.spiketrains is None:
            return None, None
        mask = (self.spiketrains['time']>=t1) & (self.spiketrains['time']<=t2)
        inwindow_label = self.spiketrains[mask]['label'].values
        inwindow_times = self.spiketrains[mask]['time'].values
        
        inwindow_selected = np.zeros(inwindow_label.shape, dtype = bool)
        
        return inwindow_times, inwindow_label, inwindow_selected

    def _plot_prediction(self,t1, t2, chunk, ind1):
        prediction = np.zeros(chunk.shape)
        
        mask = (self.spiketrains['time']>=t1) & (self.spiketrains['time']<=t2)
        spiketrains = self.spiketrains[mask]
        spike_pos = spiketrains.index-ind1
        labels = spiketrains['label'].values
        jitters = spiketrains['jitter'].values

        length = self.spikesorter.limit_right - self.spikesorter.limit_left
        catalogue = self.spikesorter.catalogue
        for i in range(spike_pos.size):
            pos = spike_pos[i] + self.spikesorter.limit_left
            if pos+length>=prediction.shape[0]: continue
            if pos<0: continue
            
            k = labels[i]
            if k<0: continue
            wf0 = catalogue[k]['center']
            wf1 = catalogue[k]['centerD']
            wf2 = catalogue[k]['centerDD']
            pred = wf0 + jitters[i]*wf1 + jitters[i]**2/2*wf2
            
            prediction[pos:pos+length, :] = pred.reshape(self.dataio.nb_channel, -1).transpose()
        
        # plotting tricks
        prediction *=self.mad
        prediction += self.med
        residuals = chunk.values - prediction
        residuals += self.med
        
        for c in range(self.dataio.nb_channel):
            if self.plot_buttons['prediction'].isChecked():
                self.curves_prediction[c].setData(chunk.index.values, prediction[:, c]*self.gains[c]+self.offsets[c])
            else:
                self.curves_prediction[c].setData([], [])
            
            if self.plot_buttons['residual'].isChecked():
                self.curves_residuals[c].setData(chunk.index.values, residuals[:, c]*self.gains[c]+self.offsets[c])
            else:
                self.curves_residuals[c].setData([], [])
                
            if not self.plot_buttons['signals'].isChecked():
                self.curves[c].setData([], [])

    def on_peak_selection_changed(self):
        selected_peaks = self.spikesorter.peak_selection[self.spikesorter.peak_selection]
        if self.params['auto_zoom_on_select'] and selected_peaks.shape[0]==1:
            seg_num, time= selected_peaks.index[0]
            if seg_num != self.seg_num:
                seg_pos = self.dataio.segments_range.index.tolist().index(seg_num)
                self.combo_seg.setCurrentIndex(seg_pos)
            self.spinbox_xsize.setValue(self.params['zoom_size'])
            self.seek(time)
        else:
            self.refresh()
    
    def item_clicked(self, plot, points):
        if self.select_button.isChecked()and len(points)==1:
            x = points[0].pos().x()
            self.spikesorter.peak_selection[:] = False
            self.spikesorter.peak_selection.loc[(self.seg_num, x)] = True
            
            self.peak_selection_changed.emit()
            self.refresh()

