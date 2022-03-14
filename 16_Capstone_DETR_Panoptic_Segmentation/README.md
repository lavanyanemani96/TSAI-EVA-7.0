# Session 16.0 Capstone


## PART I - QnA

- We take the encoded image (d x H/32 x W/32) and send it to Multi-Head Attention (FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)

The encoded image is the output of DETR’s encoder. 
 
Image (x<sub>img</sub> ∈ R<sup>3 x H x W</sup>) → 
Resnet50 penultimate layer activation (f ∈ R<sup>d' x H/32 x W/32</sup> , d'=2048) →
1x1 convolution (f ∈ R<sup>d x H/32 x W/32</sup>, d=transformer inner dimension) + pos embeddings →
Transformer encoder  (f ∈ R<sup>d x H/32 x W/32</sup>, d=transformer inner dimension)
		
The input to the ResNet50 is an image (ximg ∈ R3 x H x W) which generates a lower resolution activation map (f ∈ R<sup>d' x H/32 x W/32</sup>) as the forward pass from the ResNet50 with fixed weights by discarding the last classification layer. This activation map along with sinusoidal positional encodings becomes the input to the DETR’s encoder. The output of the encoder is concatenated to form the image (e<sub>img</sub> ∈ R<sup>d x H/32 x W/32</sup>) which is sent to the Multi-Head Attention block.


- We also send dxN Box embeddings to the Multi-Head Attention. We do something here to generate NxMxH/32xW/32 maps. (WHAT DO WE DO HERE?)

The output of the DETR’s decoder is bbox embeddings (d x N), where N = 100 is the fixed number of total instances of objects of all classes. For object detection where bbox predictions are required along with class labels, these embeddings go through FFN for final detection predictions. But for the case of panoptic segmentations, M multi-heads calculate attention scores for each N object embedding with the output of the encoder (e<sub>img</sub> ∈ R<sup>d x H/32 x W/32</sup>) described in the above answer.

- Then we concatenate these maps with Res5 Block (WHERE IS THIS COMING FROM?)

Res5 block comes from the Resnet50 encoder’s Conv5 block which is 7x7x2048 i.e. input dimensions have been reduced by 32x. This matches the encoder output shape and concatenation can be performed.

- Then we perform the above steps (EXPLAIN THESE STEPS)

1. For segmentations that follow UNet kind of approach, the first part encodes and downsamples dimensions and increases channels (till the bottom of the ‘U’). The second part then does upsampling, and decreases channels. 
2. The inputs to each upsampling step are the outputs of the previous step concatenated with the encoder output on the downsampling path. Adding the encoder outputs from downsampling path preserves finer level details to retain during upsampling. In this case the encoder is Resnet-50, and we concatenate Res5, Res4, and Res3 outputs as we move up on the other arm of ‘U’
3. Hence in the DETR modified for panoptic segmentation, as upsampling is performed, the corresponding activation from the downsampling path of the Resnet-50 encoder is concatenated 
4. Upsampling is followed by same padding convolution, normalization, ReLU
5. With 3 upsampling steps, the inputs which were at H/32,W/32 are resized to H/4,W/4 (i.e. 8x upsampling). At the same time the number of channels increases. A final extra convolution on the last step of the fpn-style cnn brings the number of channels to N. This is so that the output masks can be mutually exclusive in each of the N channels. Finally an argmax at every pixel across the stack of masks sets the pixel membership in the appropriate mask
 
Plan for training:
The training for this is split into two parts. 

1. In the first part, the DETR is trained to recognize the new type of objects and emit BB, like usual. Since the final ground truth is specified as segmentation masks, we will use cv2.getRectange(contour) API to convert that to bounding box ground truth
  A. If the classid for stuff and concrete defect are overlapping with default COCO dataset, we will need to assign new ones. Then finetune the backbone as indicated [here](https://wandb.ai/veri/detr/reports/DETR-Panoptic-segmentation-on-Cityscapes-dataset--Vmlldzo2ODg3NjE) in Step 4.1
  B. Once above model detects BB for concrete defects, we proceed to next training
2. The second part of the training, the DETR weights are frozen and we train the segmentation model for 25 epochs. 
  A. Queries corresponding to BB filtered with a high threshold (e.g. 0.85) from Step 1. are part of input to this Step 2.
  B. The fpn-style cnn is trained to convert DETR object queries + Resnet-50 encodings to emit N channel output panoptic masks - one channel for each object query. 
  C. Essentially we are doing the second arm of the UNet with ground truth being N channel masks, and loss function to be focal or dice loss. We may use dice or focal (TBD), focal helps with focussing the loss on hard to classify pixels


Contributors
-------------------------
Darshat Shah darshat@gmail.com

Lavanya Nemani lavanyanemani96@gmail.com
