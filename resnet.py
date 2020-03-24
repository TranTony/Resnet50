
from torchvision import resnet50
From torch import nn
Import torch.nn.functional as F

class Resnet_50(nn.Module):
   """
   Resnet50 class extractes regions feature   
   """

   def __init__(self, encoded_image_size=14, embed_dim=512):
       super(Resnet_50, self).__init__()
       self.embed_dim = embed_dim
       #self.decoder_dim = decoder_dim

       resnet = torchvision.models.resnet50(pretrained=True)
       self.resnet_fc = resnet50(pretrained=True)

       # Remove linear and pool layers (since we're not doing classification)
       modules = list(resnet.children())[:-2])
       self.resnet = nn.Sequential(*modules)

       # Resize image to fixed size to allow input images of variable size
       self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

       # Fine Tuning
       self.fine_tune()

   def forward(self, images):
       """
       Forward propagation.

       :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
       :return: encoded images
       """

       # output of FC
       out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
       out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
      
       # FOR MESHED MEMORY
       out = torch.flatten(out, start_dim=2, end_dim=3) # (batch_size,  2048, encoded_image_size**2)
       out = out.permute(0, 2, 1) # (batch_size, encoded_image_size ** 2, 2048)

       return out 
  
   def fine_tune(self, fine_tune=True):
       """
       Allow or prevent the computation of gradients for convolutional blocks 2 to 4 of the encoder.
       :param fine_tune: Allow?
       """
       for p in self.resnet.parameters():
           p.requires_grad = False
       # If fine-tuning, only fine-tune convolutional blocks 2 through 4
       for c in list(self.resnet.children())[5:]:
           for p in c.parameters():
               p.requires_grad = fine_tune
