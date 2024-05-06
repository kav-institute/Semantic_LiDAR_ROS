import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
import std_msgs.msg as std_msgs
import numpy as np
from ouster import pcap, client, osf
from contextlib import closing
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from sklearn.cluster import DBSCAN
from std_msgs.msg import Float32MultiArray

color_map = {
  0 : [0, 0, 0],
  1 : [245, 150, 100],
  2 : [245, 230, 100],
  3 : [150, 60, 30],
  4 : [245, 150, 100],#[180, 30, 80],
  5 : [245, 150, 100],#[255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
  9: [125,125,125],
  10: [125,125,125],#[255, 150, 255],
  11: [125,125,125],#[75, 0, 75],
  12: [125,125,125],#[75, 0, 175],
  13: [0, 200, 255],
  14: [0, 200, 255],#[50, 120, 255],
  15: [0, 175, 0],
  16: [0, 60, 135],
  17: [80, 240, 150],
  18: [0, 60, 135],#[150, 240, 255],
  19: [250, 250, 250],
  20: [0, 250, 0]
}

# Create the custom color map
custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)

for i in range(256):
    if i in color_map:
        custom_colormap[i, 0, :] = color_map[i]
    else:
        # If the index is not defined in the color map, set it to black
        custom_colormap[i, 0, :] = [0, 0, 0]
custom_colormap = custom_colormap[...,::-1]

def remove_batchnorm(model):
    # Create a new Sequential module to reconstruct the model architecture without batchnorm
    new_model = nn.Sequential()
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Skip batchnorm layers
            continue
        elif isinstance(module, nn.Sequential):
            # If the module is Sequential, recursively remove batchnorm from its children
            new_model.add_module(name, remove_batchnorm(module))
        elif name in ["0","1","2"]:
            new_model.add_module(name, remove_batchnorm(module))
        else:
            # Add other layers to the new model
            new_model.add_module(name, module)
            
    return new_model

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Define convolutional layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Define convolutional layer for attention scores
        self.attention_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, features):
        # Query, key, and value transformations
        query = self.query_conv(features)
        key = self.key_conv(features)
        value = self.value_conv(features)
        
        # Compute attention scores
        attention_scores = self.attention_conv(torch.tanh(query + key))
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to the value
        attended_features = value * attention_weights
        
        return attended_features

class SemanticNetworkWithFPN(nn.Module):#
    """
    Semantic Segmentation Network with Feature Pyramid Network (FPN) using a ResNet backbone.

    Args:
        resnet_type (str): Type of ResNet model to use. Supported types: 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
        meta_channel_dim (int): Number meta channels used in the FPN
        interpolation_mode (str): Interpolation mode used to resize the meta channels (default='nearest'). Supported types: 'nearest', 'bilinear', 'bicubic'.
        num_classes (int): Number of semantic classes
        no_bn (bool): Option to disable batchnorm (not recommended)
        attention (bool): Option to use a attention mechanism in the FPN (see: https://arxiv.org/pdf/1706.03762.pdf)
        
    """
    def __init__(self, resnet_type='resnet18', meta_channel_dim=3, interpolation_mode = 'nearest', num_classes = 3, no_bn=False, attention=True):
        super(SemanticNetworkWithFPN, self).__init__()

        self.interpolation_mode = interpolation_mode
        self.num_classes = num_classes
        self.no_bn = no_bn
        self.attention = attention
        # Load pre-trained ResNet model
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            base_channels = 512  # Number of channels in the last layer of ResNet18
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
            base_channels = 512  # Number of channels in the last layer of ResNet34
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
            base_channels = 2048  # Number of channels in the last layer of ResNet50
        elif resnet_type == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)
            base_channels = 2048  # Number of channels in the last layer of ResNet101

        else:
            raise ValueError("Invalid ResNet type. Supported types: 'resnet18', 'resnet34', 'resnet50', 'resnet101'.")

        # remove all batchnorm layers
        if self.no_bn:
            self.resnet = remove_batchnorm(self.resnet)
        
        # Modify the first convolution layer to take 1+meta_channel_dim channels
        self.meta_channel_dim = meta_channel_dim
        self.resnet.conv1 = nn.Conv2d(2 + meta_channel_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Extract feature maps from different layers of ResNet
        self.layer1 = nn.Sequential(self.resnet.conv1, self.resnet.relu, self.resnet.maxpool, self.resnet.layer1)
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        
        # Attention blocks
        self.attention4 = AttentionModule(base_channels // 2, base_channels // 2)
        self.attention3 = AttentionModule(base_channels // 4, base_channels // 4)
        self.attention2 = AttentionModule(base_channels // 8, base_channels // 8)
        self.attention1 = AttentionModule(base_channels // 16, base_channels // 16)
        

        # FPN blocks
        self.fpn_block4 = self._make_fpn_block(base_channels, base_channels // 2)
        self.fpn_block3 = self._make_fpn_block(base_channels // 2, base_channels // 4)
        self.fpn_block2 = self._make_fpn_block(base_channels // 4, base_channels // 8)
        self.fpn_block1 = self._make_fpn_block(base_channels // 8, base_channels // 16)

        # upsamle layers
        self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels // 2, out_channels=base_channels // 2, kernel_size=8, stride=8, padding=0)
        self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels // 4, out_channels=base_channels // 4, kernel_size=4, stride=4, padding=0)
        self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels // 8, out_channels=base_channels // 8, kernel_size=2, stride=2, padding=0)
        
        if self.no_bn:
            self.decoder_semantic = nn.Sequential(
                nn.Conv2d(base_channels // 2 + base_channels // 4 + base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.Conv2d(base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels // 16, self.num_classes, kernel_size=4, stride=2, padding=1),
                nn.ELU(inplace=True)
            )
        else:
            self.decoder_semantic = nn.Sequential(
                nn.Conv2d(base_channels // 2 + base_channels // 4 + base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                #nn.Conv2d(base_channels // 8 + base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 16, base_channels // 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(base_channels // 16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels // 16, self.num_classes, kernel_size=4, stride=2, padding=1),
                nn.ELU(inplace=True)
            )
    def _make_fpn_block(self, in_channels, out_channels):
        """
        Create an FPN block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Module: FPN block.
        """
        if self.no_bn:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            
    def forward(self, x, meta_channel):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing depth prediction.
        """
        
        
        # Inject meta channel before ResNet layers
        if self.meta_channel_dim > 0:
            # Resize Meta Channels
            # Downsample the meta channel
            meta_channel1 = F.interpolate(meta_channel, scale_factor=1/2, mode=self.interpolation_mode)
            meta_channel2 = F.interpolate(meta_channel, scale_factor=1/4, mode=self.interpolation_mode)
            meta_channel3 = F.interpolate(meta_channel, scale_factor=1/8, mode=self.interpolation_mode)
            x = torch.cat([x, meta_channel], dim=1)
            x1 = self.layer1(x)
            x = torch.cat([x1[:,0:-self.meta_channel_dim,...], meta_channel1], dim=1)
            x2 = self.layer2(x)
            x = torch.cat([x2[:,0:-self.meta_channel_dim,...], meta_channel2], dim=1)
            x3 = self.layer3(x)
            x = torch.cat([x3[:,0:-self.meta_channel_dim,...], meta_channel3], dim=1)
            x4 = self.layer4(x)

        else:
        
            # Encoder (ResNet)
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

        # FPN
        x4 = self.fpn_block4(x4)
        x3 = self.fpn_block3(x3)
        x2 = self.fpn_block2(x2)
        x1 = self.fpn_block1(x1)
        
        # Attention
        if self.attention:
            x4 = self.attention4(x4)
            x3 = self.attention3(x3)
            x2 = self.attention2(x2)
            x1 = self.attention1(x1)

        x4 = self.upsample_layer_x4(x4)
        x3 = self.upsample_layer_x3(x3)
        x2 = self.upsample_layer_x2(x2)

        # Concatenate feature maps
        x = torch.cat([x1, x2, x3, x4], dim=1)


        # Decoder
        x_semantics = self.decoder_semantic(x) + 1 # offset of 1 to shift elu to ]0,inf[
        
        return x_semantics

def visualize_semantic_segmentation_cv2(mask, class_colors):
    """
    Visualize semantic segmentation mask using class colors with cv2.

    Parameters:
    - mask: 2D NumPy array containing class IDs for each pixel.
    - class_colors: Dictionary mapping class IDs to BGR colors.

    Returns:
    - visualization: Colored semantic segmentation image in BGR format.
    """
    h, w = mask.shape
    visualization = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        visualization[mask == class_id] = color

    return visualization 

def build_normal_xyz(xyz, norm_factor=0.25, ksize = 3):
    '''
    @param xyz: ndarray with shape (h,w,3) containing a stagged point cloud
    @param norm_factor: int for the smoothing in Schaar filter
    '''
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    Sxx = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Sxy = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Syx = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Syy = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Szx = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Szy = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    #build cross product
    normal = -np.dstack((Syx*Szy - Szx*Syy,
                        Szx*Sxy - Szy*Sxx,
                        Sxx*Syy - Syx*Sxy))

    # normalize corss product
    n = np.linalg.norm(normal, axis=2)+1e-10
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    return normal

def point_cloud(points, parent_frame, stamp):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = std_msgs.Header(frame_id=parent_frame, stamp=stamp)

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )

class OusterPcapReaderNode(Node):
    def __init__(self):
        super().__init__('ouster_pcap_reader')
        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/ouster/point_cloud', 1)
        self.publisher_vru_array = self.create_publisher(Float32MultiArray, '/ouster/vru_array', 10)
        self.publisher_sensor_pose = self.create_publisher(Float32MultiArray, '/ouster/sensor_pose', 10)
        self.semseg_publisher = self.create_publisher(Image, '/ouster/segmentation_image', 1)

        self.metadata_path = '/home/appuser/data/Ouster/OS-2-128-992317000331-2048x10.json'
        self.pcap_path = '/home/appuser/data/Ouster/OS-2-128-992317000331-2048x10.osf'
        with open(self.metadata_path, 'r') as f:
            self.metadata = client.SensorInfo(f.read())
        self.device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")
        self.nocs_model = SemanticNetworkWithFPN(resnet_type='resnet34', meta_channel_dim=6, num_classes=20)
        self.nocs_model.load_state_dict(torch.load("/home/appuser/data//model_zoo/THAB_RN34/model_final.pth", map_location=self.device))

        # Training loop
        self.nocs_model.to(self.device)
        self.nocs_model.eval()

        # DBSCAN for clustering
        eps = 0.33  # Maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples = 32  # The number of samples in a neighborhood for a point to be considered as a core point

        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        

    def run(self):
        if isinstance(self.pcap_path, type(None)):
            load_scan = lambda: client.Scans.stream("fe80::4ce2:9d59:9f63:a9e0", 7502, complete=False, metadata=self.metadata)
        else:
            #source = pcap.Pcap(self.pcap_path, self.metadata)
            #load_scan = lambda:  client.Scans(source)
            load_scan = lambda:  osf.Scans(self.pcap_path)
        
        with closing(load_scan()) as stream:
            for scan in stream:
                # Get sensor pose from osf
                T = scan.pose[1023,...]
                msg = Float32MultiArray()
                msg.data = T.flatten().tolist()
                self.publisher_sensor_pose.publish(msg)

                xyzlut = client.XYZLut(self.metadata)

                xyz = xyzlut(scan)
                xyz = client.destagger(self.metadata, xyz)

                reflectivity_field = scan.field(client.ChanField.REFLECTIVITY)
                reflectivity_img = client.destagger(stream.metadata, reflectivity_field)
                
                range_img = np.linalg.norm(xyz,axis=-1)
                normals = build_normal_xyz(xyz)
                normal_img = np.uint8(255*(normals+1)/2)

                reflectivity_img = reflectivity_img/255.0
                reflectivity_img =  torch.as_tensor(reflectivity_img[...,None].transpose(2, 0, 1).astype("float32"))
                range_img =  torch.as_tensor(range_img[...,None].transpose(2, 0, 1).astype("float32"))
                xyz_ =  torch.as_tensor(xyz[...,0:3].transpose(2, 0, 1).astype("float32"))

                normals =  torch.as_tensor(normals.transpose(2, 0, 1).astype("float32"))
                
                range_img, reflectivity, xyz_, normals = range_img[None,...].to(self.device), reflectivity_img[None,...].to(self.device), xyz_[None,...].to(self.device), normals[None,...].to(self.device)

                outputs_semantic = self.nocs_model(torch.cat([range_img, reflectivity],axis=1), torch.cat([xyz_, normals],axis=1))

                semseg_img = torch.argmax(outputs_semantic,dim=1)
         
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                idx_VRUs = np.where(semantics_pred==6)
                prev_sem_pred = cv2.applyColorMap(np.uint8(semantics_pred), custom_colormap)
                #prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=color_map)[...,::-1]

                segment_msg = Image()
                segment_msg.header.stamp = self.get_clock().now().to_msg()
                segment_msg.header.frame_id = 'ouster_frame'
                segment_msg.height = prev_sem_pred.shape[0]
                segment_msg.width = prev_sem_pred.shape[1]
                segment_msg.encoding = 'rgb8'
                segment_msg.is_bigendian = False
                segment_msg.step = prev_sem_pred.shape[1]
                segment_msg.data = prev_sem_pred.astype(np.uint8).tobytes()
                self.semseg_publisher.publish(segment_msg)
                self.get_logger().info('Published a segmentation image')

                # # Transform point cloud to world
                # # Extend the point cloud with a fourth column of ones for homogeneous coordinates
                # xyz_h = np.concatenate((xyz, np.ones((xyz.shape[0], xyz.shape[1], 1))), axis=2)

                # # Perform the transformation by matrix multiplication
                # xyz_hT = np.matmul(T, xyz_h.reshape((-1, 4, 1))).reshape((-1, 4))[:, :3]

                # # Reshape the transformed point cloud back to its original shape
                # xyz = xyz_hT.reshape(xyz.shape)

                #Publish point cloud
                rgba = cv2.cvtColor(prev_sem_pred, cv2.COLOR_RGB2RGBA)
                pcl2 = np.concatenate([xyz,rgba/255.0],axis=-1) 
                pcl2 = pcl2.reshape(-1, pcl2.shape[-1])
                self.pointcloud_publisher.publish(point_cloud(pcl2, 'map', self.get_clock().now().to_msg()))
                self.get_logger().info('Published a point cloud')

  

def main(args=None):
    rclpy.init(args=args)
    ouster_pcap_reader_node = OusterPcapReaderNode()
    ouster_pcap_reader_node.run()
    rclpy.spin(ouster_pcap_reader_node)
    ouster_pcap_reader_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
