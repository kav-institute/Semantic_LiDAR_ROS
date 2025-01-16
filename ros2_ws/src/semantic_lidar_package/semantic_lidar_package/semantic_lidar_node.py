import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
import std_msgs.msg as std_msgs
import numpy as np
from ouster.sdk import pcap, client, osf
from contextlib import closing
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
from visualization_msgs.msg import Marker
import json

color_map = {
  0 : [0, 0, 0],
  1 : [245, 150, 100],
  2 : [245, 230, 100],
  3 : [150, 60, 30],
  4 : [180, 30, 80],
  5 : [255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
  9: [125,125,125],
  10: [255, 150, 255],
  11: [75, 0, 75],
  12: [75, 0, 175],
  13: [0, 200, 255],
  14: [50, 120, 255],
  15: [0, 175, 0],
  16: [0, 60, 135],
  17: [80, 240, 150],
  18: [150, 240, 255],
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
        backbone (str): Type of ResNet model to use. Supported types: 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
        meta_channel_dim (int): Number meta channels used in the FPN
        interpolation_mode (str): Interpolation mode used to resize the meta channels (default='nearest'). Supported types: 'nearest', 'bilinear', 'bicubic'.
        num_classes (int): Number of semantic classes
        no_bn (bool): Option to disable batchnorm (not recommended)
        attention (bool): Option to use a attention mechanism in the FPN (see: https://arxiv.org/pdf/1706.03762.pdf)
        
    """
    def __init__(self, backbone='resnet18', meta_channel_dim=3, interpolation_mode = 'nearest', num_classes = 3, attention=True, multi_scale_meta=True):
        super(SemanticNetworkWithFPN, self).__init__()

        self.backbone_name = backbone
        self.interpolation_mode = interpolation_mode
        self.num_classes = num_classes
        self.attention = attention
        self.multi_scale_meta = multi_scale_meta
        # Load pre-trained ResNet model
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            base_channel = 512  # Number of channels in the last layer of ResNet18
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            base_channel = 512  # Number of channels in the last layer of ResNet34
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            base_channel = 2048  # Number of channels in the last layer of ResNet50
            base_channels = [base_channel, base_channel // 2, base_channel // 4, base_channel // 8, base_channel // 16]
        elif backbone == 'regnet_y_400mf':
            self.backbone = models.regnet_y_400mf(pretrained=True)
            base_channels = [440, 208, 104, 48, 32]
        elif backbone == 'regnet_y_800mf':
            self.backbone = models.regnet_y_800mf(pretrained=True)
            base_channels = [784, 320, 144, 64, 32]
        elif backbone == 'regnet_y_1_6gf':
            self.backbone = models.regnet_y_1_6gf(pretrained=True)
            base_channels = [888, 336, 120, 48, 32]
        elif backbone == 'regnet_y_3_2gf':
            self.backbone = models.regnet_y_3_2gf(pretrained=True)
            base_channels = [1512, 576, 216, 72, 32]
        elif backbone == 'shufflenet_v2_x0_5':
            self.backbone = models.shufflenet_v2_x0_5(pretrained=True)
            base_channels = [1024, 192, 96, 48, 24]
        elif backbone == 'shufflenet_v2_x1_0':
            self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
            base_channels = [1024, 464, 232, 116, 24]
        elif backbone == 'shufflenet_v2_x1_5':
            self.backbone = models.shufflenet_v2_x1_5(pretrained=True)
            base_channels = [1024, 704, 352, 176, 24]
        elif backbone == 'shufflenet_v2_x2_0':
            self.backbone = models.shufflenet_v2_x2_0(pretrained=True)
            base_channels = [2048, 976, 488, 244, 112]
        elif backbone == 'squeezenet1_0':
            self.backbone = models.squeezenet1_0(pretrained=True)
            base_channels = [512, 384, 256, 256, 112]
        else:
            raise ValueError("Invalid ResNet type. Supported types: 'resnet18', 'resnet34', 'resnet50', 'regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0.")
        
        # Modify the first convolution layer to take 1+meta_channel_dim channels
        self.meta_channel_dim = meta_channel_dim

        is_shuffle = False
        is_squeeze = False
        # extract features from resnet family
        if backbone in ['resnet18', 'resnet34', 'resnet50']:
            self.backbone.conv1 = nn.Conv2d(2 + meta_channel_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)

            # Extract feature maps from different layers of ResNet
            self.stem = nn.Sequential(self.backbone.conv1, self.backbone.relu, self.backbone.maxpool)
            self.layer1 = self.backbone.layer1
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4

        elif backbone in ["squeezenet1_0"]:
            num_channels = 64 if backbone=="squeezenet1_1" else 96
            self.backbone.features[0] = nn.Conv2d(
            2 + meta_channel_dim,  # Adjust the number of input channels
            num_channels,                    # Number of output channels in the first convolutional layer
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
            self.stem = self.backbone.features[0:4]
            self.layer1 = self.backbone.features[4:6]
            self.layer2 = self.backbone.features[6:8]
            self.layer3 = self.backbone.features[8:10]
            self.layer4 = self.backbone.features[10:]
            is_squeeze = True
            
        # extract features from regnet family
        elif backbone in ['regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf']:
            self.backbone.stem[0] = nn.Conv2d(2 + meta_channel_dim, self.backbone.stem[0].out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            # Extract feature maps from different layers of RegNet
            self.stem = self.backbone.stem
            self.layer1 = self.backbone.trunk_output[0]
            self.layer2 = self.backbone.trunk_output[1]
            self.layer3 = self.backbone.trunk_output[2]
            self.layer4 = self.backbone.trunk_output[3]

        elif backbone in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"]:
            self.backbone.conv1[0] = nn.Conv2d(2 + meta_channel_dim, self.backbone.conv1[0].out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            #help_conv = nn.Conv2d(2 + meta_channel_dim, self.backbone.conv1[0].out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            # Extract feature maps using indices or named stages from ShuffleNet
            self.stem = self.backbone.conv1
            self.layer1 = self.backbone.stage2
            self.layer2 = self.backbone.stage3
            self.layer3 = self.backbone.stage4
            self.layer4 = self.backbone.conv5
            is_shuffle = True

        # Attention blocks
        self.attention4 = AttentionModule(base_channels[1], base_channels[1])
        self.attention3 = AttentionModule(base_channels[2], base_channels[2])
        self.attention2 = AttentionModule(base_channels[3], base_channels[3])
        self.attention1 = AttentionModule(base_channels[4], base_channels[4])
        

        # FPN blocks
        self.fpn_block4 = self._make_fpn_block(base_channels[0], base_channels[1])
        self.fpn_block3 = self._make_fpn_block(base_channels[1], base_channels[2])
        self.fpn_block2 = self._make_fpn_block(base_channels[2], base_channels[3])
        self.fpn_block1 = self._make_fpn_block(base_channels[3], base_channels[4])

        # upsamle layers
        if is_shuffle:
            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels[2], out_channels=base_channels[2]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels[3], out_channels=base_channels[3]//2, kernel_size=2, stride=2, padding=0)
            out_channels_upsample = base_channels[1]//4 + base_channels[2]//4 + base_channels[3]//2
        elif is_squeeze:

            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels[2], out_channels=base_channels[2]//2, kernel_size=2, stride=2, padding=0)
            self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels[3], out_channels=base_channels[3]//2, kernel_size=2, stride=2, padding=0)
            out_channels_upsample = base_channels[1]//4 + base_channels[2]//2 + base_channels[3]//2

        else:
            self.upsample_layer_x4 = nn.ConvTranspose2d(in_channels=base_channels[1], out_channels=base_channels[1]//8, kernel_size=8, stride=8, padding=0)
            self.upsample_layer_x3 = nn.ConvTranspose2d(in_channels=base_channels[2], out_channels=base_channels[2]//4, kernel_size=4, stride=4, padding=0)
            self.upsample_layer_x2 = nn.ConvTranspose2d(in_channels=base_channels[3], out_channels=base_channels[3]//2, kernel_size=2, stride=2, padding=0)
            out_channels_upsample = base_channels[1]//8 + base_channels[2]//4 + base_channels[3]//2


        self.decoder_semantic = nn.Sequential(
            nn.Conv2d(out_channels_upsample + base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels[4], base_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels[4]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels[4], self.num_classes, kernel_size=4, stride=2, padding=1),
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
        #if self.meta_channel_dim > 0:
        if self.multi_scale_meta:
            # Resize Meta Channels
            # Downsample the meta channel
            meta_channel1 = F.interpolate(meta_channel, scale_factor=1/2, mode=self.interpolation_mode)
            meta_channel2 = F.interpolate(meta_channel, scale_factor=1/4, mode=self.interpolation_mode)
            meta_channel3 = F.interpolate(meta_channel, scale_factor=1/8, mode=self.interpolation_mode)
            x = torch.cat([x, meta_channel], dim=1)
            xs = self.stem(x)
            x1 = self.layer1(xs)
            x = torch.cat([x1[:,0:-self.meta_channel_dim,...], meta_channel1], dim=1)
            x2 = self.layer2(x)
            x = torch.cat([x2[:,0:-self.meta_channel_dim,...], meta_channel2], dim=1)
            x3 = self.layer3(x)
            x = torch.cat([x3[:,0:-self.meta_channel_dim,...], meta_channel3], dim=1)
            x4 = self.layer4(x)

        else:
        
            # Encoder (ResNet)
            x = torch.cat([x, meta_channel], dim=1)
            xs = self.stem(x)
            x1 = self.layer1(xs)
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


def build_normal_xyz(xyz, norm_factor=0.25, device='cuda'):
    '''
    @param xyz: tensor with shape (h,w,3) containing a staggered point cloud
    @param norm_factor: int for the smoothing in Scharr filter
    @param device: device to move computation to (default: 'cuda')
    '''
    # Move input tensor to device
    xyz = xyz.to(device)

    x = xyz[:,0:1,...]
    y = xyz[:,1:2,...]
    z = xyz[:,2:3,...]


    # Compute partial derivatives using Scharr filter
    Sxx = F.conv2d(x, torch.tensor([[[[3, 0, -3], [10, 0, -10], [3, 0, -3]]]], dtype=torch.float32).to(device), stride=1, padding=1) / norm_factor
    Sxy = F.conv2d(x, torch.tensor([[[[3, 10, 3], [0, 0, 0], [-3, -10, -3]]]], dtype=torch.float32).to(device), stride=1, padding=1) / norm_factor

    Syx = F.conv2d(y, torch.tensor([[[[3, 0, -3], [10, 0, -10], [3, 0, -3]]]], dtype=torch.float32).to(device), stride=1, padding=1) / norm_factor
    Syy = F.conv2d(y, torch.tensor([[[[3, 10, 3], [0, 0, 0], [-3, -10, -3]]]], dtype=torch.float32).to(device), stride=1, padding=1) / norm_factor

    Szx = F.conv2d(z, torch.tensor([[[[3, 0, -3], [10, 0, -10], [3, 0, -3]]]], dtype=torch.float32).to(device), stride=1, padding=1) / norm_factor
    Szy = F.conv2d(z, torch.tensor([[[[3, 10, 3], [0, 0, 0], [-3, -10, -3]]]], dtype=torch.float32).to(device), stride=1, padding=1) / norm_factor

    # Build cross product
    normal = -torch.cat((Syx * Szy - Szx * Syy,
                         Szx * Sxy - Szy * Sxx,
                         Sxx * Syy - Syx * Sxy), dim=1)

    # Normalize cross product
    n = torch.norm(normal, dim=1, keepdim=True) + 1e-10
    normal /= n
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

    data = points.astype(dtype)
    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgb')]

    header = std_msgs.Header(frame_id=parent_frame, stamp=stamp)

    msg = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 6),
        row_step=(itemsize * 6 * points.shape[0]),
    )
    msg._data = data#.tobytes()
    return msg

class OusterPcapReaderNode(Node):
    def __init__(self):
        super().__init__('ouster_pcap_reader')
        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/ouster/semantic_point_cloud', 1)
        self.semseg_publisher = self.create_publisher(Image, '/ouster/segmentation_image', 1)
        self.mesh_publisher = self.create_publisher(Marker, "/visualization_marker", 1)

        with open("/home/appuser/data/config.json") as json_data:
            config = json.load(json_data)
            json_data.close()
        
        
        self.metadata_path = config["METADATA_PATH"]

        if not config["STREAM_SENSOR"]:
            # for recordings
            self.pcap_path = config["OSF_PATH"]
            self.stream_sensor = False
        else:
            # for sensor stream
            self.ouster_sensor_ip = config["SENSOR_IP"]
            self.ouster_sensor_port = config["SENSOR_PORT"]
            self.stream_sensor = True

        with open(self.metadata_path, 'r') as f:
            self.metadata = client.SensorInfo(f.read())

        self.device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")
        self.nocs_model = SemanticNetworkWithFPN(backbone='resnet34', meta_channel_dim=6, num_classes=20, attention=True, multi_scale_meta=True)
        self.nocs_model.load_state_dict(torch.load(config["MODEL_PATH"], map_location=self.device))

        # Training loop
        self.nocs_model.to(self.device)
        self.nocs_model.eval()



        # # load mesh
        self.mesh_msg = Marker()
        self.mesh_msg.id = 1
        self.mesh_msg.mesh_resource = 'file:////home/appuser/data/vehicle/KAV_Forschungsfahrzeug_BakedLighting.obj'
        self.mesh_msg.mesh_use_embedded_materials = True   # Need this to use textures for mesh
        self.mesh_msg.type = 10
        self.mesh_msg.header.frame_id = "map"
        self.mesh_msg.action = self.mesh_msg.ADD
        self.mesh_msg.scale.x = 1.0


        self.mesh_msg.scale.y = 1.0
        self.mesh_msg.scale.z = 1.0
        self.mesh_msg.pose.position.x = 0.0
        self.mesh_msg.pose.position.y = 0.0
        self.mesh_msg.pose.position.z = -1.75
        self.mesh_msg.pose.orientation.x = 0.0
        self.mesh_msg.pose.orientation.y = 0.0
        self.mesh_msg.pose.orientation.z = 0.0
        self.mesh_msg.pose.orientation.w = 1.0
        self.mesh_msg.color.r = 0.5
        self.mesh_msg.color.g = 0.5
        self.mesh_msg.color.b = 0.5
        self.mesh_msg.color.a = 1.0

        self.rate = self.create_rate(10) # We create a Rate object of 10Hz


        

    def run(self):
        if self.stream_sensor:
            load_scan = lambda: client.Scans.stream(self.ouster_sensor_ip, self.ouster_sensor_port, complete=False, metadata=self.metadata)
        else:
            load_scan = lambda:  osf.Scans(self.pcap_path)
        
        xyzlut = client.XYZLut(self.metadata)

        with closing(load_scan()) as stream:
            for scan in stream:
                start_time = self.get_clock().now()


                start_time_load_data = self.get_clock().now()

                xyz = xyzlut(scan)
                xyz = client.destagger(self.metadata, xyz).astype(np.float32)

                reflectivity_field = scan.field(client.ChanField.REFLECTIVITY)
                reflectivity_img = client.destagger(stream.metadata, reflectivity_field)
                
                end_time_load_data = self.get_clock().now()
                self.get_logger().info('Cycle Time Read Data: {}, {}'.format(end_time_load_data-start_time_load_data, end_time_load_data-start_time))


                start_time_inference = self.get_clock().now()
                reflectivity_img = reflectivity_img/255.0
                reflectivity_img =  torch.as_tensor(reflectivity_img[...,None].transpose(2, 0, 1).astype("float32"))

                xyz_ =  torch.as_tensor(xyz[...,0:3].transpose(2, 0, 1).astype("float32"))
                
                
                reflectivity, xyz_,  = reflectivity_img[None,...].to(self.device), xyz_[None,...].to(self.device)

                normals = build_normal_xyz(xyz_, norm_factor=0.25, device='cuda')
                range_img = torch.norm(xyz_, dim=1, keepdim=True) #+ 1e-10

                outputs_semantic = self.nocs_model(torch.cat([range_img, reflectivity],axis=1), torch.cat([xyz_, normals],axis=1))

                semseg_img = torch.argmax(outputs_semantic,dim=1)

                # Waits for everything to finish running
                torch.cuda.synchronize()

                end_time_inference = self.get_clock().now()
                self.get_logger().info('Cycle Time Inference: {} {}'.format(end_time_inference-start_time_inference, end_time_inference-start_time))

                start_time_vis= self.get_clock().now()
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                idx_VRUs = np.where(semantics_pred==6)
                prev_sem_pred = cv2.applyColorMap(np.uint8(semantics_pred), custom_colormap)
                end_time_vis= self.get_clock().now()
                self.get_logger().info('Cycle Time Vis: {} {}'.format(end_time_vis-start_time_vis, end_time_vis-start_time))

                start_time_img_pub= self.get_clock().now()
                prev_sem_pred_ = cv2.resize(prev_sem_pred,(1024,64), interpolation = cv2.INTER_NEAREST)
                segment_msg = Image()
                segment_msg.header.stamp = self.get_clock().now().to_msg()
                segment_msg.header.frame_id = 'ouster_frame'
                segment_msg.height = prev_sem_pred_.shape[0]
                segment_msg.width = prev_sem_pred_.shape[1]
                segment_msg.encoding = 'rgb8'
                segment_msg.is_bigendian = True
                segment_msg.step = prev_sem_pred_.shape[1]
                segment_msg.data = prev_sem_pred_.astype(np.uint8).tobytes()
                self.semseg_publisher.publish(segment_msg)
                #self.get_logger().info('Published a segmentation image')
                end_time_img_pub= self.get_clock().now()
                self.get_logger().info('Cycle Time Publish Image: {} {}'.format(end_time_img_pub-start_time_img_pub, end_time_img_pub-start_time))



                #Publish point cloud
                start_time_pc_header = self.get_clock().now()
                start_time_pc_pub= self.get_clock().now()
                #rgba = cv2.cvtColor(prev_sem_pred, cv2.COLOR_RGB2RGBA)
                pcl2 = np.concatenate([xyz,prev_sem_pred/255.0],axis=-1)
                pcl2 = pcl2.reshape(-1, pcl2.shape[-1])


                
                self.pointcloud_publisher.publish(point_cloud(pcl2, 'map', self.get_clock().now().to_msg()))
                #self.get_logger().info('Published a point cloud')
                
                end_time_pc_pub= self.get_clock().now()
                end_time = self.get_clock().now()
                self.get_logger().info('Cycle Time Publish PC: {} {}'.format(end_time_pc_pub-start_time_pc_pub, end_time-start_time))

                
                # load mesh
                self.get_logger().info("Publishing the mesh topic. Use RViz to visualize.")
                #self.mesh_msg.header.stamp = self.get_clock().now().to_msg()
                self.mesh_publisher.publish(self.mesh_msg)
  

def main(args=None):
    rclpy.init(args=args)
    ouster_pcap_reader_node = OusterPcapReaderNode()
    ouster_pcap_reader_node.run()
    rclpy.spin(ouster_pcap_reader_node)
    ouster_pcap_reader_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
