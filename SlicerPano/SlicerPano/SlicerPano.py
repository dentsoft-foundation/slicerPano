import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

# additional libraries
import numpy as np

#
# SlicerPano
#

class SlicerPano(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "SlicerPano" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Dentistry"]
    self.parent.dependencies = []
    self.parent.contributors = ["Georgi Talmazov (Dental Software Foundation)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """The pantomograph and implant planning extension for 3D Slicer."""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """This extension is developed and maintained by the Dental Software Foundation (https://dentsoft.foundation/).""" # replace with organization, grant and thanks.

#
# SlicerPanoWidget
#

class SlicerPanoWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    # Flythough variables
    self.transform = None
    self.path = None

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Path collapsible button
    pathCollapsibleButton = ctk.ctkCollapsibleButton()
    pathCollapsibleButton.text = "Path"
    self.layout.addWidget(pathCollapsibleButton)

    # Layout within the path collapsible button
    pathFormLayout = qt.QFormLayout(pathCollapsibleButton)

    # Input fiducials node selector
    inputFiducialsNodeSelector = slicer.qMRMLNodeComboBox()
    inputFiducialsNodeSelector.objectName = 'inputFiducialsNodeSelector'
    inputFiducialsNodeSelector.toolTip = "Select a fiducial list to define control points for the path."
    inputFiducialsNodeSelector.nodeTypes = ['vtkMRMLMarkupsFiducialNode', 'vtkMRMLAnnotationHierarchyNode', 'vtkMRMLFiducialListNode']
    inputFiducialsNodeSelector.noneEnabled = False
    inputFiducialsNodeSelector.addEnabled = False
    inputFiducialsNodeSelector.removeEnabled = False
    inputFiducialsNodeSelector.connect('currentNodeChanged(bool)', self.enableOrDisableCreateButton)
    pathFormLayout.addRow("Input Fiducials:", inputFiducialsNodeSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        inputFiducialsNodeSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Initialize button - essentially building the path projection model
    InitializeButton = qt.QPushButton("Initialize")
    InitializeButton.toolTip = "Create the path."
    InitializeButton.enabled = False
    pathFormLayout.addRow(InitializeButton)
    InitializeButton.connect('clicked()', self.onInitializeButtonClicked)


    # Flythrough collapsible button
    flythroughCollapsibleButton = ctk.ctkCollapsibleButton()
    flythroughCollapsibleButton.text = "Options"
    flythroughCollapsibleButton.enabled = False
    self.layout.addWidget(flythroughCollapsibleButton)

    # Layout within the Flythrough collapsible button
    flythroughFormLayout = qt.QFormLayout(flythroughCollapsibleButton)

    # Frame slider
    frameSlider = ctk.ctkSliderWidget()
    frameSlider.connect('valueChanged(double)', self.frameSliderValueChanged)
    frameSlider.decimals = 0
    flythroughFormLayout.addRow("Frame:", frameSlider)

    # Build Pantomograph button
    PantomographButton = qt.QPushButton("Build Pantomograph")
    PantomographButton.toolTip = "Build pantomograph from fiducial path model."
    flythroughFormLayout.addRow(PantomographButton)
    PantomographButton.connect('clicked()', self.onPantomographButtonToggled)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Set local var as instance attribute
    self.inputFiducialsNodeSelector = inputFiducialsNodeSelector
    self.InitializeButton = InitializeButton
    self.flythroughCollapsibleButton = flythroughCollapsibleButton
    self.frameSlider = frameSlider
    self.PantomographButton = PantomographButton

    inputFiducialsNodeSelector.setMRMLScene(slicer.mrmlScene)

    ########################################################################################
    XML_layout = """
      <layout type="horizontal" split="true">
      <item splitSize="250">
          <layout type="horizontal" split="true">
            <item>
              <view class="vtkMRMLSliceNode" singletontag="Transverse">
              <property name="orientation" action="default">Reformat</property>
              <property name="viewlabel" action="default">TR</property>
              <property name="viewcolor" action="default">#000000</property>
              </view>
            </item>
          </layout>
       </item>
       <item splitSize="500">
        <layout type="vertical" split="true">
         <item>
          <view class="vtkMRMLSliceNode" singletontag="Red">
           <property name="orientation" action="default">Axial</property>
           <property name="viewlabel" action="default">R</property>
           <property name="viewcolor" action="default">#F34A33</property>
          </view>
         </item>
         <item>
          <view class="vtkMRMLViewNode" singletontag="1">
           <property name="viewlabel" action="default">1</property>
          </view>
         </item>
        </layout>
       </item>
       <item splitSize="500">
        <layout type="vertical" split="true">
         <item>
          <view class="vtkMRMLSliceNode" singletontag="Yellow">
           <property name="orientation" action="default">Sagittal</property>
           <property name="viewlabel" action="default">Y</property>
           <property name="viewcolor" action="default">#EDD54C</property>
          </view>
         </item>
         <item>
          <view class="vtkMRMLSliceNode" singletontag="Green">
           <property name="orientation" action="default">Coronal</property>
           <property name="viewlabel" action="default">G</property>
           <property name="viewcolor" action="default">#6EB04B</property>
          </view>
         </item>
        </layout>
       </item>
      </layout>

    """
    # Built-in layout IDs are all below 100, so you can choose any large random number
    # for your custom layout ID.
    customLayoutId=501
    volumeNode = slicer.util.getNode('1*')
    #Creating a dummy slice node for computing
    # Create slice node (this automatically creates a slice widget)
    sliceNodeCompute = slicer.mrmlScene.CreateNodeByClass('vtkMRMLSliceNode')
    sliceNodeCompute.SetName("PanoCompute")
    sliceNodeCompute.SetLayoutName("PanoCompute")
    sliceNodeCompute.SetLayoutLabel("PC")
    sliceNodeCompute.SetOrientation("Sagittal")
    sliceNodeCompute = slicer.mrmlScene.AddNode(sliceNodeCompute)

    # Select background volume
    sliceLogic = slicer.app.applicationLogic().GetSliceLogic(sliceNodeCompute)
    sliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())

    # Get the automatically created slice widget
    lm=slicer.app.layoutManager()
    sliceWidget=lm.viewWidget(sliceNodeCompute)
    controller = lm.sliceWidget("PanoCompute").sliceController()
    controller.setSliceVisible(True)
    sliceWidget.setParent(None)
    # Show slice widget
    
    #sliceNodeCompute.SetSliceVisible(0)
    #slicer.app.layoutManager().sliceWidget('PanoCompute').mrmlSliceNode().SetDimensions(250, 500, 1)


    lm.layoutLogic().GetLayoutNode().AddLayoutDescription(customLayoutId, XML_layout)                                         

    # Switch to the new custom layout 
    lm.setLayout(customLayoutId)

    #Re-load primary volume across all slice nodes
    
    slicer.util.setSliceViewerLayers(background=volumeNode)

    #enables all slices' views in 3D view
    #layoutManager = slicer.app.layoutManager()
    #for sliceViewName in layoutManager.sliceViewNames():
    #  controller = layoutManager.sliceWidget(sliceViewName).sliceController()
    #  controller.setSliceVisible(True)

    #enable Transverse slice view in 3D viewer
    transSliceNode = lm.sliceWidget("Transverse").sliceController()
    transSliceNode.setSliceVisible(True)

    sliceNodeCompute.SetMappedInLayout(1)


    ########################################################################################


  def enableOrDisableCreateButton(self):
    """Connected to both the fiducial and camera node selector. It allows to
    enable or disable the 'create path' button."""
    self.InitializeButton.enabled = self.inputFiducialsNodeSelector.currentNode() is not None

  def onInitializeButtonClicked(self):
    """Connected to 'create path' button. It allows to:
      - compute the path
      - create the associated model"""

    fiducialsNode = self.inputFiducialsNodeSelector.currentNode()
    #print "Calculating Path..."
    result = EndoscopyComputePath(fiducialsNode)
    #print "-> Computed path contains %d elements" % len(result.path)

    #print "Create Model..."
    self.model = EndoscopyPathModel(result.path, fiducialsNode)
    #print "-> Model created"

    # Update frame slider range
    self.frameSlider.maximum = len(result.path) - 2

    # Update flythrough variables
    self.transform = self.model.transform
    self.pathPlaneNormal = self.model.planeNormal
    self.path = result.path


    
    # Enable / Disable flythrough button
    self.flythroughCollapsibleButton.enabled = len(result.path) > 0

    #FOV = self.model.reslice_on_path(self.path[1], self.path[2], 1)
    #self.model.build_pano(self.path, slicer.util.getNode('1*'), FOV)

    
  def frameSliderValueChanged(self, newValue):
    ##print "frameSliderValueChanged:", newValue
    self.flyTo(newValue)

  def onPantomographButtonToggled(self):
    FOV = self.model.reslice_on_path(self.path[1], self.path[2], 1)
    self.model.build_pano(self.path, slicer.util.getNode('1*'), FOV)

  def flyToNext(self):
    currentStep = self.frameSlider.value
    nextStep = currentStep + self.skip + 1
    if nextStep > len(self.path) - 2:
      nextStep = 0
    self.frameSlider.value = nextStep

  def flyTo(self, f):
    """ Apply the fth step in the path to the global camera"""
    if self.path:
      f = int(f)
      p = self.path[f]
      
      FOV = self.model.reslice_on_path(self.path[f], self.path[f+1], f)
      #self.model.build_pano(self.path, slicer.util.getNode('1*'), FOV)
      #self.model.test()
      
      


class EndoscopyComputePath:
  """Compute path given a list of fiducials.
  A Hermite spline interpolation is used. See http://en.wikipedia.org/wiki/Cubic_Hermite_spline

  Example:
    result = EndoscopyComputePath(fiducialListNode)
    #print "computer path has %d elements" % len(result.path)

  """

  def __init__(self, fiducialListNode, dl = 0.5):
    import numpy
    self.dl = dl # desired world space step size (in mm)
    self.dt = dl # current guess of parametric stepsize
    self.fids = fiducialListNode

    # hermite interpolation functions
    self.h00 = lambda t: 2*t**3 - 3*t**2     + 1
    self.h10 = lambda t:   t**3 - 2*t**2 + t
    self.h01 = lambda t:-2*t**3 + 3*t**2
    self.h11 = lambda t:   t**3 -   t**2

    # n is the number of control points in the piecewise curve

    if self.fids.GetClassName() == "vtkMRMLAnnotationHierarchyNode":
      # slicer4 style hierarchy nodes
      collection = vtk.vtkCollection()
      self.fids.GetChildrenDisplayableNodes(collection)
      self.n = collection.GetNumberOfItems()
      if self.n == 0:
        return
      self.p = numpy.zeros((self.n,3))
      for i in range(self.n):
        f = collection.GetItemAsObject(i)
        coords = [0,0,0]
        f.GetFiducialCoordinates(coords)
        self.p[i] = coords
    elif self.fids.GetClassName() == "vtkMRMLMarkupsFiducialNode":
      # slicer4 Markups node
      self.n = self.fids.GetNumberOfFiducials()
      n = self.n
      if n == 0:
        return
      # get fiducial positions
      # sets self.p
      self.p = numpy.zeros((n,3))
      for i in range(n):
        coord = [0.0, 0.0, 0.0]
        self.fids.GetNthFiducialPosition(i, coord)
        self.p[i] = coord
    else:
      # slicer3 style fiducial lists
      self.n = self.fids.GetNumberOfFiducials()
      n = self.n
      if n == 0:
        return
      # get control point data
      # sets self.p
      self.p = numpy.zeros((n,3))
      for i in range(n):
        self.p[i] = self.fids.GetNthFiducialXYZ(i)

    # calculate the tangent vectors
    # - fm is forward difference
    # - m is average of in and out vectors
    # - first tangent is out vector, last is in vector
    # - sets self.m
    n = self.n
    fm = numpy.zeros((n,3))
    for i in range(0,n-1):
      fm[i] = self.p[i+1] - self.p[i]
    self.m = numpy.zeros((n,3))
    for i in range(1,n-1):
      self.m[i] = (fm[i-1] + fm[i]) / 2.
    self.m[0] = fm[0]
    self.m[n-1] = fm[n-2]

    self.path = [self.p[0]]
    self.calculatePath()

  def calculatePath(self):
    """ Generate a flight path for of steps of length dl """
    #
    # calculate the actual path
    # - take steps of self.dl in world space
    # -- if dl steps into next segment, take a step of size "remainder" in the new segment
    # - put resulting points into self.path
    #
    n = self.n
    segment = 0 # which first point of current segment
    t = 0 # parametric current parametric increment
    remainder = 0 # how much of dl isn't included in current step
    while segment < n-1:
      t, p, remainder = self.step(segment, t, self.dl)
      if remainder != 0 or t == 1.:
        segment += 1
        t = 0
        if segment < n-1:
          t, p, remainder = self.step(segment, t, remainder)
      self.path.append(p)

  def point(self,segment,t):
    return (self.h00(t)*self.p[segment] +
              self.h10(t)*self.m[segment] +
              self.h01(t)*self.p[segment+1] +
              self.h11(t)*self.m[segment+1])

  def step(self,segment,t,dl):
    """ Take a step of dl and return the path point and new t
      return:
      t = new parametric coordinate after step
      p = point after step
      remainder = if step results in parametic coordinate > 1.0, then
        this is the amount of world space not covered by step
    """
    import numpy.linalg
    p0 = self.path[self.path.__len__() - 1] # last element in path
    remainder = 0
    ratio = 100
    count = 0
    while abs(1. - ratio) > 0.05:
      t1 = t + self.dt
      pguess = self.point(segment,t1)
      dist = numpy.linalg.norm(pguess - p0)
      ratio = self.dl / dist
      self.dt *= ratio
      if self.dt < 0.00000001:
        return
      count += 1
      if count > 500:
        return (t1, pguess, 0)
    if t1 > 1.:
      t1 = 1.
      p1 = self.point(segment, t1)
      remainder = numpy.linalg.norm(p1 - pguess)
      pguess = p1
    return (t1, pguess, remainder)


class EndoscopyPathModel:
  """Create a vtkPolyData for a polyline:
       - Add one point per path point.
       - Add a single polyline
  """
  def __init__(self, path, fiducialListNode):

    fids = fiducialListNode
    scene = slicer.mrmlScene

    points = vtk.vtkPoints()
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)

    lines = vtk.vtkCellArray()
    polyData.SetLines(lines)
    linesIDArray = lines.GetData()
    linesIDArray.Reset()
    linesIDArray.InsertNextTuple1(0)

    polygons = vtk.vtkCellArray()
    polyData.SetPolys( polygons )
    idArray = polygons.GetData()
    idArray.Reset()
    idArray.InsertNextTuple1(0)

    for point in path:
      pointIndex = points.InsertNextPoint(*point)
      linesIDArray.InsertNextTuple1(pointIndex)
      linesIDArray.SetTuple1( 0, linesIDArray.GetNumberOfTuples() - 1 )
      lines.SetNumberOfCells(1)

    import vtk.util.numpy_support as VN
    pointsArray = VN.vtk_to_numpy(points.GetData())
    self.planePosition, self.planeNormal = self.planeFit(pointsArray.T)

    # Create model node
    model = slicer.vtkMRMLModelNode()
    model.SetScene(scene)
    model.SetName(scene.GenerateUniqueName("Path-%s" % fids.GetName()))
    model.SetAndObservePolyData(polyData)

    # Create display node
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetColor(1,1,0) # yellow
    modelDisplay.SetScene(scene)
    scene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())

    # Add to scene
    scene.AddNode(model)

    # Camera cursor
    sphere = vtk.vtkSphereSource()
    sphere.Update()

    # Create model node
    cursor = slicer.vtkMRMLModelNode()
    cursor.SetScene(scene)
    cursor.SetName(scene.GenerateUniqueName("Cursor-%s" % fids.GetName()))
    cursor.SetPolyDataConnection(sphere.GetOutputPort())

    # Create display node
    cursorModelDisplay = slicer.vtkMRMLModelDisplayNode()
    cursorModelDisplay.SetColor(1,0,0) # red
    cursorModelDisplay.SetScene(scene)
    scene.AddNode(cursorModelDisplay)
    cursor.SetAndObserveDisplayNodeID(cursorModelDisplay.GetID())

    # Add to scene
    scene.AddNode(cursor)

    # Create transform node
    transform = slicer.vtkMRMLLinearTransformNode()
    transform.SetName(scene.GenerateUniqueName("Transform-%s" % fids.GetName()))
    scene.AddNode(transform)
    cursor.SetAndObserveTransformNodeID(transform.GetID())

    self.transform = transform

  # source: http://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
  def planeFit(self, points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    import numpy as np
    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]

  # Inputting origin and normal point (+1 step from origin on model path), then fitting a poly-line, obtain derivative, build normal line from tangent, obtain coordinate = this yields 3 point coordinate for RasByNTP
  def reslice_on_path(self, p0, pN, file_num):
    fx=np.poly1d(np.polyfit([p0[0],pN[0]],[p0[1],pN[1]], 1))
    fdx = np.polyder(fx)
    normal_line = lambda x: (-1/fdx(p0[0]))*(x-p0[0])+p0[1]
    t=np.array([p0[0]+1,normal_line(p0[0]+1),p0[2]], dtype='f')
    t=t-p0
    n=pN-p0
    t.astype(float)
    n.astype(float)
    p0.astype(float)
    sliceNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeTransverse")
    sliceNode.SetSliceToRASByNTP(n[0], n[1], n[2], t[0], t[1], t[2], p0[0], p0[1], p0[2], 0)

    sliceToRas = sliceNode.GetSliceToRAS()
    if (sliceToRas.GetElement(1, 0) > 0 and sliceToRas.GetElement(1, 2) > 0) or (sliceToRas.GetElement(0, 2) > 0 and sliceToRas.GetElement(1, 0) < 0):
      transform = vtk.vtkTransform()
      transform.SetMatrix(sliceToRas)
      transform.RotateZ(180)
      #transform.RotateY(180)
      sliceToRas.DeepCopy(transform.GetMatrix())
      sliceNode.UpdateMatrices()

    #rescaling dimensions to zoom in using slice node's aspect ratio
    x = 50 # lower number = zoom-in default 50 for pano ~10
    y = x * sliceNode.GetFieldOfView()[1] / sliceNode.GetFieldOfView()[0]
    z = sliceNode.GetFieldOfView()[2]
    sliceNode.SetFieldOfView(x,y,z)
    sliceNode.Modified()

    widget = slicer.app.layoutManager().sliceWidget("Transverse")
    view = widget.sliceView()
    view.forceRender()

    # saving an image from the slice view
    #image = qt.QPixmap.grabWidget(view).toImage()
    #image.convertToFormat(qt.QImage.Format_Grayscale8)
    #ptr = image.bits()
    #arr = np.array(ptr).reshape(image.height(), image.width(), 4)
    #print arr[500][100]
    #image.save('C:/Users/talmazovg/Downloads/temp/img-%03d.png' % file_num)
    return [x,y,z] #FOV zoomed in further by factor of 5 to improve visualization, consider implenting image enhancement

  def build_pano(self, model_nodes, volumeNode, FOV):
    curveNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode')
    curveNode.CreateDefaultDisplayNodes()
    curveNode.GetCurveGenerator().SetNumberOfPointsPerInterpolatingSegment(2) # add more curve points between control points than the default 10
    for node in model_nodes:
      curveNode.AddControlPoint(vtk.vtkVector3d(node[0],	node[1],	node[2])) #default node[2], 0 centers in the z-plane?

    sliceNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodePanoCompute') #sliceNodeCompute
    rotationAngleDeg = 180
    sliceNode.SetFieldOfView(FOV[0], FOV[1], FOV[2]) # zoom in

    appLogic = slicer.app.applicationLogic()
    sliceLogic = appLogic.GetSliceLogic(sliceNode)
    sliceLayerLogic = sliceLogic.GetBackgroundLayer()
    reslice = sliceLayerLogic.GetReslice()
    reslicedImage = vtk.vtkImageData()

    # Straightened volume (useful for example for visualization of curved vessels)
    straightenedVolume = slicer.modules.volumes.logic().CloneVolume(volumeNode, 'straightened')

    # Capture a number of slices orthogonal to the curve and append them into a volume.
    # sliceToWorldTransform = curvePointToWorldTransform * RotateZ(rotationAngleDeg)
    curvePointToWorldTransform = vtk.vtkTransform()
    sliceToWorldTransform = vtk.vtkTransform()
    sliceToWorldTransform.Concatenate(curvePointToWorldTransform)
    sliceToWorldTransform.RotateZ(rotationAngleDeg)
    sliceNode.SetXYZOrigin(0,0,0)
    numberOfPoints = curveNode.GetCurvePointsWorld().GetNumberOfPoints()
    append = vtk.vtkImageAppend()

    for pointIndex in range(numberOfPoints):
        #print(pointIndex)
        curvePointToWorldMatrix = vtk.vtkMatrix4x4()
        curveNode.GetCurvePointToWorldTransformAtPointIndex(pointIndex, curvePointToWorldMatrix)
        curvePointToWorldTransform.SetMatrix(curvePointToWorldMatrix)
        sliceToWorldTransform.Update()
        sliceNode.GetSliceToRAS().DeepCopy(sliceToWorldTransform.GetMatrix())
        sliceNode.UpdateMatrices()
        slicer.app.processEvents()
        tempSlice = vtk.vtkImageData()
        tempSlice.DeepCopy(reslice.GetOutput())
        append.AddInputData(tempSlice)

    append.SetAppendAxis(2)
    append.Update()
    straightenedVolume.SetAndObserveImageData(append.GetOutput())

    # Create panoramic volume by mean intensity projection along an axis of the straightened volume
    #import numpy as np
    panoramicVolume = slicer.modules.volumes.logic().CloneVolume(straightenedVolume, 'panoramic')
    panoramicImageData = panoramicVolume.GetImageData()
    straightenedImageData = straightenedVolume.GetImageData()
    panoramicImageData.SetDimensions(straightenedImageData.GetDimensions()[2], straightenedImageData.GetDimensions()[1], 1)
    panoramicImageData.AllocateScalars(straightenedImageData.GetScalarType(), straightenedImageData.GetNumberOfScalarComponents())
    panoramicVolumeArray = slicer.util.arrayFromVolume(panoramicVolume)
    straightenedVolumeArray = slicer.util.arrayFromVolume(straightenedVolume)
    panoramicVolumeArray[0, :, :] = np.flip(straightenedVolumeArray.mean(2).T)
    slicer.util.arrayFromVolumeModified(panoramicVolume)
    panoramicVolume.SetSpacing(1.0, 1.0, 1.0) # just approximate spacing (would need to properly compute from FOV and image size)
    sliceNode.SetOrientationToAxial()
    #sliceNode.SetSliceVisible(1)
    #slicer.util.setSliceViewerLayers(background=panoramicVolume, label="PC", fit=True)
    panoramicVolume.GetDisplayNode().AutoWindowLevelOn()
    #sliceLogic = slicer.app.applicationLogic().GetSliceLogic(sliceNode)
    sliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(panoramicVolume.GetID())




    


#
# SlicerPanoLogic
#

class SlicerPanoLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : imageThreshold, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('SlicerPanoTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class SlicerPanoTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SlicerPano1()

  def test_SlicerPano1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = SlicerPanoLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')


