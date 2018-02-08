video-super-resolution : A convolutional neural network implementation with Tensorflow
====================

using Python 3.6.3 :: Anaconda custom (64-bit) , Tensorflow 1.1.0

Cut video : ffmpeg -ss 00:00:30.0 -i input.avi -c copy -t 00:00:10.0 output.avi (https://superuser.com/questions/138331/using-ffmpeg-to-cut-up-video)

Split video into images : ffmpeg -i file.avi -r 50/1 $output%03d.bmp (https://stackoverflow.com/questions/10957412/fastest-way-to-extract-frames-using-ffmpeg)

Train module : (video-super-resolution)$ python main.py --stride=14 --is_train=True

Test module : (video-super-resolution)$ python main.py --stride=21 --is_train=False

