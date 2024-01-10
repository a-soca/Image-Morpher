import cv2 # Computer vision library (opencv)
import sys # Used to close all windows when exiting the program
import numpy as np # Used for matrices

############################################################################

# Coordinates are stored here from the leftClick function and moved to different matrices after input is complete
inputCoordinates = []

# Landmark locations are stored here for easy modification
landmarks = ["chin", "mouth", "top of philtrum", "nose tip", "eye", "top of forehead", "point between the top of forehead and top of skull", "top of skull", "point where skull begins to slope back", "back of skull", "top of neck (back)", "bottom of neck (back)", "front of neck", "top of throat"]

############################################################################
# Functions
############################################################################

# Function to detect mouse click location
def leftClick(event, x, y, flags, parameters):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(displayedImage, (x,y), 5, (255,100,255), 2) # Adds point to image where user clicked
        # Store click coordinates
        inputCoordinates.append((x,y))
        if len(inputCoordinates) < len(landmarks): # Check if there are more landmarks required and ask user to click if so
            print("Please click the " + landmarks[len(inputCoordinates)])

# Function to calculate the triangular segments from the landmarks
def triangulate(imagePath, coordinates):
    image = cv2.imread(imagePath) # Import the source image
    size = image.shape # Find the dimensions
    boundary = (0, 0, size[1], size[0]) # Create boundary based on size
    subdivisions = cv2.Subdiv2D(boundary) # Subdivide this boundary ready for triangulation

    subdivisions.insert(coordinates) # Insert the landmark coordinates into the subdivisions matrix

    triangles = subdivisions.getTriangleList() # Create triangles from provided subdivisions 
    triangles = np.array(triangles, dtype=np.float32) # Cast into 32 bit integer values
    return triangles


# Function to find the indeces of the coordinate points for a triangle within an array
def findIndeces(triangles, coordinates):
    indeces = []
    coordinates = np.array(coordinates)
    for t in triangles:
        # Converts each triangle into 3 vertices
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # Finds the position where points match the source coordinates and converts them into indeces
        pt1Index = np.where(coordinates == pt1) 
        pt1Index = pt1Index[0][0]
        pt2Index = np.where(coordinates == pt2)
        pt2Index = pt2Index[0][0]
        pt3Index = np.where(coordinates == pt3)
        pt3Index = pt3Index[0][0]
        # Stores the indeces
        if pt1Index is not None and pt2Index is not None and pt3Index is not None: # If the vertices were in the coordinates array
            triangle = [pt1Index, pt2Index, pt3Index] # Create triangle index reference
            indeces.append(triangle) # Add to triangle indeces
    return indeces # Returns a list of the indeces which lead to each point in the triangles so the source and target triangulations can be matched

############################################################################
# Main Program
############################################################################

# SOURCE IMAGE PROCESSING
# Importing image
displayedImage = cv2.imread("source.png")

# Create window to display images
cv2.namedWindow('Preview')
cv2.setMouseCallback('Preview', leftClick) # Bind click callback to window

# Display instructions in console
print("\n----------------------------\n\tSource Image\n----------------------------\n")
print("In the image window, click the following landmarks (Going around the head in one direction)")
print("Please click the " + landmarks[0]) # Print the first landmark location to click

# Display Source image and update with clicked points
while True:
    cv2.imshow('Preview', displayedImage) # Redraw the image in the window to show updates
    if ~cv2.waitKey(1) or len(inputCoordinates) == len(landmarks): # If the user presses a key to escape or the landmarks have been input successsfully, exit the loop
        break

# Reset and prepare for target image analysis
cv2.destroyAllWindows()

# Check to see if the user did not enter all points and exit
if len(inputCoordinates) < len(landmarks):
    sys.exit()

# Store collected data points in sourceX and sourceY
sourceCoordinates = inputCoordinates

# Clear the input arrays for reuse
inputCoordinates = []

# Create new window with target image and reassign the user click callback to this window
cv2.namedWindow('Preview')
cv2.setMouseCallback('Preview', leftClick)
displayedImage = cv2.imread("target.png")

# TARGET IMAGE PROCESSING
# Display instructions in console
print("\n----------------------------\n\tTarget Image\n----------------------------\n")
print("In the image window, click the following landmarks (Going around the face in one direction)")
print("Please click the " + landmarks[0]) # Print the first landmark location to click

# Display Source image and update with clicked points
while True:
    cv2.imshow('Preview', displayedImage)
    if ~cv2.waitKey(1) or len(inputCoordinates) == len(landmarks):
        break

# Close the window displaying the image
cv2.destroyAllWindows()

# Check to see if the user did not enter all points and exit
if len(inputCoordinates) < len(landmarks):
    sys.exit()
        
# Store the input coordinates in targetX and targetY for readability
targetCoordinates = inputCoordinates

############################################################################
# Morphing
############################################################################

# Perform triangulation (Delaunay Triangulation) on images
trianglesSource = triangulate("source.png", sourceCoordinates)
sourceIndeces = findIndeces(trianglesSource, sourceCoordinates)

trianglesTarget = [] # Initialise array to store the target image triangle coordinates
for trianglePoints in sourceIndeces: # For all the indeces previously found, store the points in an array
    pt1 = targetCoordinates[trianglePoints[0]]
    pt2 = targetCoordinates[trianglePoints[1]]
    pt3 = targetCoordinates[trianglePoints[2]]

    trianglePoints = pt1 + pt2 + pt3
    trianglesTarget.append(trianglePoints)
    
trianglesTarget = np.array(trianglesTarget, dtype=np.float32) # Convert this matrix into a numpy array

source = cv2.imread('source.png') # Open the source image
# Check to see if the images are facing different directions based on the landmark selections of user
front = landmarks.index("nose tip") # By checking the index of the elements we are interested in, more landmark points can be added to the array without editing any other code.
back = landmarks.index("back of skull")
if(sourceCoordinates[front][1] < sourceCoordinates[back][1]): # If source image is facing left
    if(targetCoordinates[front][1] > targetCoordinates[back][1]): # AND if target image is facing right
        displayedImage = cv2.flip(displayedImage, 1) # Flip the source image 
elif(targetCoordinates[front][1] < targetCoordinates[back][1]): # Otherwise, if the target image is facing left (previous if statement means source image must be facing right)
        displayedImage = cv2.flip(displayedImage, 1) # Flip the source image 

target = cv2.imread('target.png') # Open the target image
morphed = np.zeros(target.shape, dtype = np.float32) # Initialise the canvas for the morphed image (black)
h, w = target.shape[:2] # Find the dimensions for the warp canvas
for i in range(0, len(trianglesTarget)): # For all the triangles to morph, loop
    src = trianglesSource[i].reshape(3,2) # reshape the source triangle coordinates into a matrix and store as src
    dest = trianglesTarget[i].reshape(3,2) # reshape the targettriangle coordinates into a matrix and store as dest
    
    # To increase efficiency, the source image could be cropped to the boundaries of the src triangle here

    # Generate Target Mask
    destinationMask = np.zeros(target.shape[:2]) # Initialise target mask with size of canvas
    destTriangleMask = np.array(dest, dtype=int) # Create array of points for the mask triangle to be drawn from
    cv2.fillPoly(destinationMask, pts=[destTriangleMask], color=(255, 255, 255)) # Add a white triangle with the same size and position as the target triangle to the black mask
    kernel = np.ones((2,1),np.uint8) # Create kernel to shrink mask edges
    destinationMask = cv2.erode(destinationMask, kernel, iterations = 1) # Shrink edges to prevent lines between segments from addition process
    destinationMask = destinationMask.astype(np.int8) # Convert mask to 8 bit integer
    

    # Warp and Mask
    transformMatrix = cv2.getAffineTransform(src, dest) # Find the transformation matrix from the triangle points
    warpedTriangle = cv2.warpAffine(source, transformMatrix, (w, h)) # Warp the image
    warpedTriangle = cv2.bitwise_and(warpedTriangle, warpedTriangle, mask=destinationMask) # Mask the warped image into a triangle
    

    morphed = cv2.add(morphed, warpedTriangle.astype(np.float32)) # Add this triangle to the final output canvas


############################################################################
# Final output
############################################################################

morphed = cv2.add(morphed, target.astype(np.float32)) # add the target image to the morphed source image
cv2.namedWindow('Final') # Create window to display result
morphed = morphed/255 # as imshow must have pixel values between 0 and 1, the values from 0-255 should be divided by 255
print("Press any key while the image window is in focus to exit")
cv2.imshow('Final', morphed) # Display the image in the window 
cv2.waitKey(0) # Waits for user to dismiss window
cv2.imwrite('Processed.png', morphed*255) # Exports the image (imwrite needs values from 0-255 so we must remultiply the values)
print("\n\nImage saved as Processed.png\n")

sys.exit()

