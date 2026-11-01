# Work in progress notes

Here are some notes that i made while doing this assessment. Mostly to explain my progress and decisions.

# Timeline

1. First I wanted to lay the basic codebase structure. Create the web server, routes and services. A structure like any other DI project available. 
    * I created the controllers at `routes` dir
    * A detection service that concerns finding and listing all checkboxes
    * and a classifier service that takes these found checkboxes and determines wether they are checked or not.

2. Set up an initial detection service that it's using vanilla openCV. It turns the image into grayscale, binarizes it and then finds contours. Based on all contours, it would try to find squares:
    * check if contour has area of square (height*width)
    * check if contour has 4 corners
    * checks if contour has "checkbox size" (a small number that depends on the size of the page. Totally arbitrary but worked for that time)
    * make sure element is not rectangular (width must be roughly equal to height)

3. A classifier was needed, and at the very beginning, I believe it was good to set a baseline. So an excellent naive classifier for this is to assume all checkboxes are unchecked. Think about it: if there is a checkbox in a document, its at least a yes or no question (so a 50-50 guess); there are even more times when its more than just a yes or no question, and in there you usually choose only one, so if there are 5 options, its a 80% guess of hitting the mark. I used this classifier as the BASE CASE for any other classifier.

4. Having created the detection and classifier services, I set up the tests to get results faster. Time limit is a week, and this is needed more than anything. At first, I used the images on the assessment PDF, which I carefully labelled and with LabelStudio. They have 286 checkboxes which is a good enough metric for small tests. My naive classifier was actually pretty good: 68% (beats 50% at least). 

5. Having set up testing, I needed to optimize. First thing was setting up a real classifier. I first looked for a pretrained classifier online, and found this one: [wendys-llc/checkbox-classifier](https://huggingface.co/wendys-llc/checkbox-classifier). It seemingly was able to classify checkboxes, to an advertised 95% accuracy. Small note was that this was trained on web UI checkboxes, not documents. The result was that this classifier beat my base case classifier by a very slight margin (70% vs 68%). It also took waaay too long to process any image. It was a good learning moment.

6. As existing pretrained classifier didn't match this domain, I went on to make my own small one. For this I found the following dataset: [meghnagera15/checkbox-cropped-binary](https://huggingface.co/datasets/meghnagera15/checkbox-cropped-binary/tree/main), which already provided multiple labelled fixed-sized images of checkboxes. I took these images and made my own CNN classifier based on this. This classifier has ~100k parameters, and achieved 94,8% accuracy on the split part of the checkbox dataset and 99,8% accuracy on the images at `/data`.

> [!NOTE]
> When can a checkbox be considered as `is_checked`?
> In my experience, it really depends on the context and the rules of each form. Forms usually tell you to `cross` or `tick` checkboxes, and some times even `fill` them. In some scenarios, like standardized tests, when you crossed a wrong box and want to correct it, they tell you to fill the box and then cross or tick the right one. So the final `checked` answer depends more on the context. 
> As there are no rules at play here or no subclassification of wether a box is crossed, ticked, filled or just empty, this model assumes a box `is_checked` when the box is crossed, ticked or filled.

7. Having achieved a pretty good classifier, now the test results showed 99,8% accuracy on classification and 67% accuracy on detection. So it was time to tackle a more hardened detection method. By taking a look at the detection, I noticed that while empty checkboxes are easily found, crossed checkboxes are much more difficult to properly identify, as the crosses they contain usually break the contouring heuristics I use to identify them. I've tried multiple approaches such as adjusting filters, and even trying to filter off diagonals and just keep straight lines to easily identify the box of the checkbox. The problem is that they throw off a lot of false positives.

8. I've found the following dataset ["roboflow.com/cogniquest-idc1w/checkboxes-xtv3p"](https://universe.roboflow.com/cogniquest-idc1w/checkboxes-xtv3p) which contains almost 2000 labelled scanned document images with checkboxes. I used this to train a YOLOv8 object detection model.