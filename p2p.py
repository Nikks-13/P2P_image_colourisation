import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

clr_path = "/kaggle/input/landscape-image-colorization/landscape Images/color"
gry_path = "/kaggle/input/landscape-image-colorization/landscape Images/gray"

import os

clr_img_path = []
gry_img_path = []

for img_path in os.listdir(clr_path) :
    clr_img_path.append(os.path.join(clr_path, img_path))
    
for img_path in os.listdir(gry_path) :
    gry_img_path.append(os.path.join(gry_path, img_path))

clr_img_path.sort()
gry_img_path.sort()

from PIL import Image
from tensorflow.keras.utils import img_to_array
X = []
y = []

for i in range(5000) :
    
    img1 = cv2.cvtColor(cv2.imread(clr_img_path[i]), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(gry_img_path[i]), cv2.COLOR_BGR2RGB)
    
    y.append(img_to_array(Image.fromarray(cv2.resize(img1,(128,128)))))
    X.append(img_to_array(Image.fromarray(cv2.resize(img2,(128,128)))))

X = np.array(X)
y = np.array(y)

plt.figure(figsize = (10,50))

i = 0

while i < 20:
    
    x = np.random.randint(0,3000)
    
    plt.subplot(10, 2, i+1)
    plt.imshow(X[x]/255.0,'gray')
    plt.axis('off')
    plt.title('Gray Image')
    
    plt.subplot(10, 2, i+2)
    plt.imshow(y[x]/ 255.0)
    plt.axis('off')
    plt.title('ColorImage')
    i += 2
    
plt.show()

X = (X/127.5) - 1
y = (y/127.5) - 1

print(X.shape)
print(y.shape)

print(f'Minimum of X : {X.min()}')
print(f'Maximum of X : {X.max()}')

print(f'Minimum of y : {y.min()}')
print(f'Maximum of y : {y.max()}')

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, shuffle = False)

print(X_train.shape)
print(y_train.shape)

print(X_valid.shape)
print(y_valid.shape)

from tensorflow_addons.layers import SpectralNormalization
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization

init = RandomNormal(mean = 0.0, stddev = 0.02)

def d_block (x_input, filters, strides, padding, batch_norm, inst_norm) :
    
    x = Conv2D(filters, (4, 4),
               strides=strides,
               padding=padding,
               use_bias= False,
               kernel_initializer = init)(x_input)
    
    '''
    
    '''
    
    if batch_norm == True :
        x = BatchNormalization   ()(x)
    if inst_norm  == True :
        x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x


def u_block (x, skip, filters, strides, padding, batch_norm, inst_norm) :
    
    x = Conv2DTranspose(filters, (4, 4),
                        strides=strides,
                        padding=padding,
                        use_bias= False,
                        kernel_initializer = init)(x)
    
    '''
     normalizations applied here as well.
    '''
    
    if batch_norm == True :
        x = BatchNormalization   ()(x)
    if inst_norm  == True :
        x = InstanceNormalization()(x)
    x = ReLU()(x)
    conc_x = Concatenate()([x , skip])
    
    return conc_x

def PatchGAN (image_shape) :
    
    genI = Input(shape =  image_shape)
    tarI = Input(shape =  image_shape)
    conc = Concatenate()([genI, tarI])
    
    c064 = d_block(conc, 2**6, 2, 'same', False, False)
    c128 = d_block(c064, 2**7, 2, 'same', False, True )
    c256 = d_block(c128, 2**8, 2, 'same', True , False)
    
    temp = ZeroPadding2D()(c256)
    
    c512 = d_block(temp, 2**9, 1,'valid', True , False)
    
    temp = ZeroPadding2D()(c512)
    
    c001 = Conv2D(2**0, (4,4), strides=1, padding = 'valid', activation = 'sigmoid', kernel_initializer=init)(temp)
    
    model = Model(inputs = [genI, tarI], outputs = c001)
    return model

d_model = PatchGAN((128,128,3,))
d_model.summary()

from keras.utils import plot_model
plot_model(d_model, './d_model.png', show_shapes = True)

def mod_Unet () :
    
    srcI = Input(shape = (128,128,3,))
    
    # Contracting path
    
    c064 = d_block(srcI, 2**6, 2, 'same', False, False) # _______________________.
    c128 = d_block(c064, 2**7, 2, 'same', True , False) # ____________________.  .
    c256 = d_block(c128, 2**8, 2, 'same', True , False) # _________________.  .  .
    c512 = d_block(c256, 2**9, 2, 'same', True , False) # ______________.  .  .  .
    d512 = d_block(c512, 2**9, 2, 'same', True , False) # ___________.  .  .  .  .
    e512 = d_block(d512, 2**9, 2, 'same', True , False) # ________.  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    # Bottleneck layer                                            .  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    f512 = d_block(e512, 2**9, 2, 'same', True , False) #         .  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    # Expanding  path                                             .  .  .  .  .  .
                                                        #         .  .  .  .  .  .
    u512 = u_block(f512, e512, 2**9, 2, 'same', True, False)# ____.  .  .  .  .  .
    u512 = u_block(u512, d512, 2**9, 2, 'same', True, False)# _______.  .  .  .  .
    u512 = u_block(u512, c512, 2**9, 2, 'same', True, False)# __________.  .  .  .
    u256 = u_block(u512, c256, 2**8, 2, 'same', True, False)# _____________.  .  .
    u128 = u_block(u256, c128, 2**7, 2, 'same', True, False)# ________________.  .
    u064 = u_block(u128, c064, 2**6, 2, 'same', False, True)# ___________________.
    
    genI = Conv2DTranspose(3, (4,4), strides = 2, padding = 'same', activation = 'tanh', kernel_initializer = init)(u064)
    
    model = Model(inputs = srcI, outputs = genI)
    return model

g_model = mod_Unet()
g_model.summary()

plot_model(g_model, './g_model.png', show_shapes = True)

LAMBDA = 100
BATCH_SIZE = 16
BUFFER_SIZE  = 400

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)
valid_dataset = valid_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE)

gen0 = mod_Unet()
dis0 = PatchGAN((128,128,3,)) # (W//1) x (H//1)
dis1 = PatchGAN((64, 64, 3,)) # (W//2) x (H//2)
dis2 = PatchGAN((32, 32, 3,)) # (W//4) x (H//4)

bin_entropy = keras.losses.BinaryCrossentropy(from_logits = True)

def gen_loss (dis_gen_output, target_image, gen_output) :
    
    ad_loss = bin_entropy(tf.ones_like (dis_gen_output) ,  dis_gen_output)
    l1_loss = tf.reduce_mean(tf.abs(tf.subtract(target_image,gen_output)))
    

    total_loss = ad_loss + (LAMBDA*l1_loss)
    
    return total_loss, ad_loss, l1_loss

def dis_loss (dis_gen_output, dis_tar_output) :
    
    gen_loss = bin_entropy(tf.zeros_like(dis_gen_output), dis_gen_output)
    tar_loss = bin_entropy(tf.ones_like (dis_tar_output), dis_tar_output)
    
    total_dis_loss = gen_loss + tar_loss
    return total_dis_loss
img  = cv2.imread('../input/landscape-image-colorization/landscape Images/color/12.jpg')   #change here
img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (128,128))
a128 = img_to_array(Image.fromarray(img))

a128/= 255.0

a064 = cv2.resize(a128, (64,64))
a032 = cv2.resize(a064, (32,32))

plt.figure(figsize = (20,20))

plt.subplot(1,3,1)
plt.imshow(a128)
plt.title('128x128', fontsize = 20)
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(a064)
plt.title('64 x 64', fontsize = 20)
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(a032)
plt.title('32 x 32', fontsize = 20)
plt.axis('off')

g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5, beta_2=0.999)
d0optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5, beta_2=0.999)
d1optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5, beta_2=0.999)
d2optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5, beta_2=0.999)

@tf.function
def train_on_batch (b_w_image, tar_image) :
    
    with tf.GradientTape(persistent = True) as  g :
        
        '''
        Image Tensors
        '''
        gen_image = gen0(b_w_image, training=True)
        
        # 128x128
        dis_tar_output_128 = dis0([b_w_image, tar_image], training = True)
        dis_gen_output_128 = dis0([b_w_image, gen_image], training = True)
        
        
        tar_image_128 = tar_image
        gen_image_128 = gen_image
        
        tar_image = tf.image.resize(tar_image, [64,64])
        b_w_image = tf.image.resize(b_w_image, [64,64])
        gen_image = tf.image.resize(gen_image, [64,64])
        
        # 064x064
        dis_tar_output_064 = dis1([b_w_image, tar_image], training = True)
        dis_gen_output_064 = dis1([b_w_image, gen_image], training = True)
        
        tar_image_064 = tar_image
        gen_image_064 = gen_image
        
        tar_image = tf.image.resize(tar_image, [32,32])
        b_w_image = tf.image.resize(b_w_image, [32,32])
        gen_image = tf.image.resize(gen_image, [32,32])
        
        # 032x032
        dis_tar_output_032 = dis2([b_w_image, tar_image], training = True)
        dis_gen_output_032 = dis2([b_w_image, gen_image], training = True)
        
        tar_image_032 = tar_image
        gen_image_032 = gen_image
        
        '''
        LOSS
        '''
        
        # 128x128
        g_loss_128, _, _ = gen_loss(dis_gen_output_128, tar_image_128, gen_image_128)
        d_loss_128 = dis_loss(dis_gen_output_128, dis_tar_output_128)
        
        # 064x064
        g_loss_064, _, _ = gen_loss(dis_gen_output_064, tar_image_064, gen_image_064)
        d_loss_064 = dis_loss(dis_gen_output_064, dis_tar_output_064)
        
        # 032x032
        g_loss_032, _, _ = gen_loss(dis_gen_output_032, tar_image_032, gen_image_032)
        d_loss_032 = dis_loss(dis_gen_output_032, dis_tar_output_032)
        
        
        g_total_loss = g_loss_128 + g_loss_064 + g_loss_032
        d_total_loss = d_loss_128 + d_loss_064 + d_loss_032
    
    # compute gradients
    g_gradients = g.gradient(g_total_loss, gen0.trainable_variables) # generatorLoss
    
    d0gradients = g.gradient(d_loss_128, dis0.trainable_variables)   # dis loss 128
    d1gradients = g.gradient(d_loss_064, dis1.trainable_variables)   # dis loss 064
    d2gradients = g.gradient(d_loss_032, dis2.trainable_variables)   # dis loss 032
    
    
    # apply gradient descent
    g_optimizer.apply_gradients(zip(g_gradients, gen0.trainable_variables))
    
    d0optimizer.apply_gradients(zip(d0gradients, dis0.trainable_variables))
    d1optimizer.apply_gradients(zip(d1gradients, dis1.trainable_variables))
    d2optimizer.apply_gradients(zip(d2gradients, dis2.trainable_variables))

for global_b_w_image, global_tar_image in train_dataset.take(1) :
    pass

idx=0
def save_image(image, name):
    # convert tensor to numpy array
    image = ((image[0] + 1.0) / 2.0).numpy() * 255.0
    # save image to file
    plt.imsave(name, image.astype('uint8'))
    
def fig(b_w_image, gen_image, tar_image, idx):
    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow((b_w_image[0] + 1.0) / 2.0)
    plt.title('B&W Image', fontsize=20)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow((gen_image[0] + 1.0) / 2.0)
    plt.title('Gen Image', fontsize=20)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow((tar_image[0] + 1.0) / 2.0)
    plt.title('Ori Image', fontsize=20)
    plt.axis('off')
    
    # save images separately
    save_image(b_w_image, f'bw_{idx}.png')
    save_image(gen_image, f'gen_{idx}.png')
    save_image(tar_image, f'tar_{idx}.png') 
    plt.show()

def fig1(b_w_image, gen_image, tar_image):
    
    plt.figure(figsize=(10, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow((b_w_image[0] + 1.0) / 2.0)
    plt.title('B&W Image', fontsize=20)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow((gen_image[0] + 1.0) / 2.0)
    plt.title('Gen Image', fontsize=20)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow((tar_image[0] + 1.0) / 2.0)
    plt.title('Ori Image', fontsize=20)
    plt.axis('off')
    

    plt.show()

def fit (EPOCHS = 200) :
    
    for epoch in range(EPOCHS) :
        
        print(f'Epoch {epoch} out of {EPOCHS}')
        
        for n, (b_w_image, tar_image) in train_dataset.enumerate() :
            if n ==  265 :
                print('#....End')
            if n%20 == 0 :
                print('#',end='')
            train_on_batch(b_w_image, tar_image)
        
        if epoch%3  == 0 :
            global_gen_image = gen0(global_b_w_image,training = True)
            fig1(global_b_w_image, global_gen_image, global_tar_image)

fit(EPOCHS = 100)

for idx, (b_w_image, tar_image) in enumerate(valid_dataset.take(700)):
    # generate image using generator model
    gen_image = gen0(b_w_image, training=True)
    # plot and save images
    fig(b_w_image, gen_image, tar_image, idx)

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(target_image_path, generated_image_path):
    # Load images
    target_image = cv2.imread(target_image_path)
    generated_image = cv2.imread(generated_image_path)
    # Convert images to grayscale
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

# Calculate SSIM
    ssim_score = ssim(target_gray, generated_gray)

# Calculate PSNR
    mse = np.mean((target_image - generated_image) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

# Calculate MAE
    original_img = target_image.astype(np.float32)
    colorized_img = generated_image.astype(np.float32)
    mae = np.mean(np.abs(original_img- colorized_img))

# Calculate accuracy
    threshold = 10
    # Calculate difference between images
    diff = np.abs(original_img - colorized_img)
    # Calculate maximum absolute difference for each pixel
    max_diff = np.max(diff, axis=2)   
    # Count number of pixels with maximum difference below threshold
    num_correct_pixels = np.sum(max_diff <= threshold)
    # Calculate accuracy as percentage of correct pixels
    accuracy = num_correct_pixels / (original_img.shape[0] * original_img.shape[1])
    accuracy=accuracy*100
   

    return ssim_score, psnr, mae, accuracy

ssim_scores = []
psnr_scores = []
mae_scores = []
accuracy_scores = []

for i in range(0, 10):
    generated_image_path = f'/kaggle/working/gen_{i}.png'
    target_image_path = f'/kaggle/working/tar_{i}.png'
    
    ssim_score, psnr_score, mae_score, accuracy_score = calculate_metrics(target_image_path, generated_image_path)
    
    ssim_scores.append(ssim_score)
    psnr_scores.append(psnr_score)
    mae_scores.append(mae_score)
    accuracy_scores.append(accuracy_score)
    
    print(f"Metrics for image {i}: SSIM={ssim_score:.4f}, PSNR={psnr_score:.4f}, MAE={mae_score:.4f}, ACC={accuracy_score:.2f}%")

print("Average metrics:")
print(f"SSIM={np.mean(ssim_scores):.4f}, PSNR={np.mean(psnr_scores):.4f}, MAE={np.mean(mae_scores):.4f}, ACC={np.mean(accuracy_scores):.2f}%")

gen0.save('gen_model.h5')
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

# Load the saved model
model=load_model('gen_model.h5')


input_paths = ['/kaggle/working/bw_23.png', '/kaggle/input/landscape-image-colorization/landscape Images/gray/5016.jpg','/kaggle/input/landscape-image-colorization/landscape Images/gray/5008.jpg',
               '/kaggle/working/bw_43.png','/kaggle/working/bw_35.png' ,'/kaggle/working/bw_28.png','/kaggle/working/bw_26.png','/kaggle/working/bw_4.png']

# Loop through each input image and generate the output image
for input_path in input_paths:
    # Preprocess the input image
    image = cv2.imread(input_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    image = tf.expand_dims(image, axis=0)

    # Generate the output image
    output = model(image)

    # Postprocess the output image
    output = tf.squeeze(output, axis=0)
    output = tf.clip_by_value(output, 0, 1)
    output = tf.image.convert_image_dtype(output, dtype=tf.uint8)
    output = tf.image.encode_jpeg(output)

    # Save the output image
    output_path = '/kaggle/working/output_image' + input_path.split('/')[-1]
    with open(output_path, 'wb') as f:
        f.write(output.numpy())

    # Display the input and output images
    input_image = cv2.cvtColor(image[0].numpy(), cv2.COLOR_RGB2BGR)
    output_image = cv2.imread(output_path)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Output Image')
    plt.axis('off')
    plt.show()
