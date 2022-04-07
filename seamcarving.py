import cv2
import numpy as np
import sys

def derivatives(img):
    """Calculate initial energy of an image using the first derivative in the x and y directions.
    Args:
        img (np.array): RGB color image, 3-dimensional numpy array of shape (height, width, channels)
    Returns:
        energy (np.array): 3-dimensional numpy array of shape (height, width, channels) representing the initial energy of the image

    """
    b, g, r = np.moveaxis(img, 2, 0)
    b_dy = cv2.Sobel(b, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_CONSTANT)
    b_dx = cv2.Sobel(b, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_CONSTANT)
    g_dy = cv2.Sobel(g, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_CONSTANT)
    g_dx = cv2.Sobel(g, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_CONSTANT)
    r_dy = cv2.Sobel(r, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_CONSTANT)
    r_dx = cv2.Sobel(r, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_CONSTANT)
    dx = np.dstack([abs(b_dx), abs(g_dx), abs(r_dx)])
    dy = np.dstack([abs(b_dy), abs(g_dy), abs(r_dy)])
    energy = dx + dy

    return energy


def backward_energy_table(energy):
    """Calculate energy table using a backward cumulative energy function .
    Args:
        energy (np.array): 3-dimensional numpy array of shape (height, width, channels) representing the initial energy of the image
    Returns:
        backtrack (np.array): 3-dimensional numpy array of shape (height, width, channels) representing the backward cumulative energy of the image

    """
    b_energy, g_energy, r_energy = np.moveaxis(energy, 2, 0)
    energies = [b_energy, g_energy, r_energy]
    tables = []
    for e in range(len(energies)):
        backtrack_tbl = np.zeros((len(energies[e]), len(energies[e][0])))
        # intialize the table with the initial energies to work from
        backtrack_tbl[0] = energies[e][0]
        # the value of each cell in the table of cumulative energies is determined by finding the minimum cumulative energy from 
        # the cumulative energies in the table directly above the current cell (i.e., upper left, direct upper, and upper right cells)
        for i in range(1, len(energies[e]), 1):
            for j in range(len(energies[e][i])):
                if j == 0:
                    backtrack_tbl[i][j] = energies[e][i][j] + min(backtrack_tbl[i-1][j], backtrack_tbl[i-1][j+1])
                elif j == (len(energies[e][0])-1):
                    backtrack_tbl[i][j] = energies[e][i][j] + min(backtrack_tbl[i-1][j-1], backtrack_tbl[i-1][j])
                else:
                    backtrack_tbl[i][j] = energies[e][i][j] + min(backtrack_tbl[i-1][j-1], backtrack_tbl[i-1][j], backtrack_tbl[i-1][j+1])
        tables.append(backtrack_tbl)
    backtrack = np.dstack([tables[0], tables[1], tables[2]])
    return backtrack

def forward_energy_table(energy):
    """Calculate energy table using a forward cumulative energy function .
    Args:
        energy (np.array): 3-dimensional numpy array of shape (height, width, channels) representing the initial energy of the image
    Returns:
        backtrack (np.array): 3-dimensional numpy array of shape (height, width, channels) representing the forward cumulative energy of the image

    """
    tables = []
    b, g, r = np.moveaxis(energy, 2, 0)
    im = [b, g, r]
    for e in range(len(im)):
        padded = im[e].astype(np.float64)
        backtrack_tbl = np.zeros((len(padded), len(padded[0])))
        # for forward energy, we don't initialize with the initial energies but with the inital pixel values of the image
        backtrack_tbl[0] = padded[0]
        # The forward approach for creating a cumulative energy table is similar to the backward approach, except instead 
        # of just taking the minimum of the cumulative energies from the row before for the current cumulative energy, 
        # there is a cost to add to each cumulative energy that comes from the image pixel intensities
        for i in range(1, len(padded), 1):
            for j in range(len(padded[i])):
                if j == 0:
                    # c is the cost from the image pixel intensities
                    c = abs(padded[i-1][j] - padded[i][j+1])
                    backtrack_tbl[i][j] = min((backtrack_tbl[i-1][j] + c), (backtrack_tbl[i-1][j+1] + c))
                elif j == (len(padded[i])-1):
                    c = abs(padded[i-1][j] - padded[i][j-1])
                    backtrack_tbl[i][j] = min((backtrack_tbl[i-1][j-1] + c), (backtrack_tbl[i-1][j] + c))
                else:
                    cl = abs(padded[i][j+1] - padded[i][j-1]) + abs(padded[i-1][j] - padded[i][j-1])
                    cu = abs(padded[i][j+1] - padded[i][j-1])
                    cr = abs(padded[i][j+1] - padded[i][j-1]) + abs(padded[i-1][j] - padded[i][j+1])
                    backtrack_tbl[i][j] = min((backtrack_tbl[i-1][j-1] + cl), (backtrack_tbl[i-1][j] + cu), backtrack_tbl[i-1][j+1] + cr)
        tables.append(backtrack_tbl)
    backtrack = np.dstack([tables[0], tables[1], tables[2]])
    return backtrack


def find_start(backtrack_tbl):
    """Find the place in the bottom row of the energy table with the minimum cumulative energy to know where to being your seam.
    Args:
        backtrack (np.array): 3-dimensional numpy array of shape (height, width, channels) representing the cumulative energy of the image
    Returns:
        possible_starts (list): a list of possible starting points for the best seam, based on the minimum cumulative energy

    """
    back_1, back_2, back_3 = np.moveaxis(backtrack_tbl, 2, 0)
    tables = [back_1, back_2, back_3]
    possible_starts = [[] for x in range(3)]
    for b in range(len(tables)):
        energy_and_coords = []
        for i in range(len(tables[b][-1])):
            energy_and_coords.append((i, tables[b][-1][i]))
        energy_and_coords.sort(key=lambda x: x[1])
        for i in range(len(energy_and_coords)):
            if energy_and_coords[i][1] == energy_and_coords[0][1]:
                possible_starts[b].append(energy_and_coords[i][0])
    return possible_starts


def find_seam(image, backtrack_tbl, start_js):
    """Find the whole seam by backtracking through the cumulative energy table (starts at the minimum energy in the bottom row, 
        looks back at the energies from the previous row directly above, to the upper left, and to the upper right of the minimum 
        energy and finds the minimum out of those and continues in that pattern until it has reached the first row, keeping track of 
        the cells that are part of the seam), and then determines which seam is the best to remove by calculating which seam has the 
        lowest energy when all energies of the cells that make up the seam are added together.
    Args:
        image (np.array): RGB color image, 3-dimensional numpy array of shape (height, width, channels)
        backtrack_tbl (np.array): 3-dimensional numpy array of shape (height, width, channels) representing the cumulative energy of the image
        start_js (list): a list of possible starting points for the best seam, based on the minimum cumulative energy
    Returns:
        seam (list): list of coordinates of cells in the image array that make up the seam to remove

    """
    start_i = len(backtrack_tbl)-1
    total_possible_sums = []
    total_possible_seams = []
    back_1, back_2, back_3 = np.moveaxis(backtrack_tbl, 2, 0)
    tables = [back_1, back_2, back_3]
    for p in range(len(start_js)):
        # find all possible seams because there can be duplicate minimum energies to start from
        possible_seams = [0 for x in range(len(start_js[p]))]
        seams = [[] for x in range(len(start_js[p]))]
        for s in range(len(start_js[p])):
            start_j = start_js[p][s]
            for i in range(len(tables[p]), 0, -1):
                if start_j == 0:
                    checks = [tables[p][i-1][start_j], tables[p][i-1][start_j+1]]
                    coords = [(i-1, start_j), (i-1, start_j+1)]
                elif start_j == (len(tables[p][0])-1):
                    checks = [tables[p][i-1][start_j-1], tables[p][i-1][start_j]]
                    coords = [(i-1, start_j-1), (i-1, start_j)]
                else:
                    checks = [tables[p][i-1][start_j-1], tables[p][i-1][start_j], tables[p][i-1][start_j+1]]
                    coords = [(i-1, start_j-1), (i-1, start_j), (i-1, start_j+1)]
                min_next_val = min(checks)
                for c in range(len(checks)):
                    if min_next_val == checks[c]:
                        possible_seams[s] += min_next_val
                        start_i = coords[c][0]
                        start_j = coords[c][1]
                seams[s].append([start_i, start_j])
        total_possible_sums.append(possible_seams)
        total_possible_seams.append(seams)
    sums = []
    seams = []
    for i in range(len(total_possible_seams)):
        for j in range(len(total_possible_seams[i])):
            sums.append(total_possible_sums[i][j])
            seams.append(total_possible_seams[i][j])
    # find the optimal seam by finding the seam with the minimum total energy
    seam = seams[sums.index(min(sums))]
    seam.reverse()
    b, g, r = np.moveaxis(image, 2, 0)
    channels = [b, g]
    for c in range(len(channels)):
        for i in range(len(seam)):
            channels[c][seam[i][0], seam[i][1]] = 0
    for i in range(len(seam)):
        r[seam[i][0], seam[i][1]] = 255
    return seam
    

def remove_seam(img, seam):
    """Remove the optimal seam found from the image.
    Args:
        img (np.array): RGB color image, 3-dimensional numpy array of shape (height, width, channels)
        seam (list): list of coordinates of cells in the image array that make up the seam to remove
    Returns:
        new (np.array): the new image (3-dimensional numpy array of shape (height, widht-1, channels)) with the seam removed

    """
    b, g, r = np.moveaxis(img, 2, 0)
    channels = [b, g, r]
    resized = [[] for x in range(3)]
    for c in range(len(channels)):
        resized[c] = np.zeros((len(channels[c]), (len(channels[c][0])-1)))
        for i in range(len(channels[c])):
            resized[c][i] = np.delete(channels[c][i], seam[i][1])
    new = np.dstack([resized[0], resized[1], resized[2]])
    return new


def add_seam(img, seam):
    """Color the seams to be removed in red and show on original image.
    Args:
        img (np.array): RGB color image, 3-dimensional numpy array of shape (height, width, channels)
        seam (list): list of coordinates of cells in the image array that make up the seam to add to image in red
    Returns:
        new (np.array): the original image with the seams to be removed colored in red

    """
    b, g, r = np.moveaxis(img, 2, 0)
    channels = [b, g, r]
    resized = [[] for x in range(3)]
    for c in range(len(channels)):
        resized[c] = np.zeros((len(channels[c]), (len(channels[c][0])+1)))
        for i in range(len(channels[c])):
            resized[c][i] = np.insert(channels[c][i], seam[i][1], channels[c][i][seam[i][1]-1])
    channels = [resized[0], resized[1]]
    for c in range(len(channels)):
        for i in range(len(seam)):
            channels[c][seam[i][0]][seam[i][1]] = 0
    for i in range(len(seam)):
        resized[2][seam[i][0]][seam[i][1]] = 255
    new = np.dstack([resized[0], resized[1], resized[2]])
    return new


def enlarge(img, seam, seams):
    """Stretch image by adding optimal seam (duplicate the pixel values in the seam to create new column of pixels).
    Args:
        img (np.array): RGB color image, 3-dimensional numpy array of shape (height, width, channels)
        seam (list): list of coordinates of cells in the image array that make up the seam to add to image in red
        seams (list): list of lists containing all optimal seams to remove
    Returns:
        new (np.array): the enlarged image with the optimal seam added

    """
    b, g, r = np.moveaxis(img, 2, 0)
    channels = [b, g, r]
    resized = [[] for x in range(3)]
    for c in range(len(channels)):
        resized[c] = np.zeros((len(channels[c]), (len(channels[c][0])+1)))
        for i in range(len(channels[c])):
            resized[c][i] = np.insert(channels[c][i], seam[i][1], channels[c][i][seam[i][1]-1])
    for i in range(len(seams)):
        for j in range(len(seams[i])):
            if (seam[j][1] < seams[i][j][1] < len(resized[c][0])-2):
                seams[i][j][1] += 2
            else:
                pass
    new = np.dstack([resized[0], resized[1], resized[2]])
    return new, seams


if __name__ == "__main__":
    image_file = sys.argv[1]
    num_seams = int(sys.argv[2])
    type_seam_carving = sys.argv[3]
    energy = sys.argv[4]
    output = sys.argv[5]

    img = cv2.imread(image_file)
    seams = []
    im = np.copy(img)
    im2 = np.copy(img)

    # Remove seams using the backward cumulative energy function
    if type_seam_carving == 'shrink' and energy == 'backward':
        add_seams = sys.argv[6]
        for i in range(num_seams):
            # Step 1, find the initial energy of the image
            e = derivatives(im)
            # Step 2, create an energy table
            backtrack = backward_energy_table(e)
            # Step 3, find the place in the bottom row of the energy table with the minimum cumulative
            # energy to know where to being your seam
            start = find_start(backtrack)
            # Step 4, From the minimum cumulative energy point, work backwards up the energy table to 
            # determine the best seam to remove (the seam with the lowest total energy)
            seam = find_seam(im, backtrack, start)
            # Step 5, Add the best seams to remove to the list of seams you want to remove
            seams.append(seam)
            # Step 6, Remove the seams from the image
            new = remove_seam(im, seam)
            im = new
        # Add in the seams in red that you want to remove to the original image
        if add_seams == 'yes':
            for i in range(num_seams-1, 0, -1):
                new = add_seam(im2, seams[i])
                im2 = new
            cv2.imwrite(output, im2)
        else:
            cv2.imwrite(output, im)
    # Remove seams using the forward cumulative energy function
    elif type_seam_carving == 'shrink' and energy == 'forward':
        add_seams = sys.argv[5]
        for i in range(num_seams):
            e = derivatives(im)
            backtrack = forward_energy_table(e)
            start = find_start(backtrack)
            seam = find_seam(im, backtrack, start)
            seams.append(seam)
            new = remove_seam(im, seam)
            im = new
        if add_seams == 'yes':
            for i in range(num_seams-1, 0, -1):
                new = add_seam(im2, seams[i])
                im2 = new
            cv2.imwrite(output, im2)
        else:
            cv2.imwrite(output, im)
    # Add seams using the backward cumulative energy function
    elif type_seam_carving == 'stretch' and energy == 'backward':
        for i in range(num_seams):
            e = derivatives(im)
            backtrack = backward_energy_table(e)
            start = find_start(backtrack)
            seam = find_seam(im, backtrack, start)
            seams.append(seam)
            new = remove_seam(im, seam)
            im = new
        for i in range(len(seams)):
            new, adjust_seams = enlarge(im2, seams[i], seams)
            seams = adjust_seams
            im2 = new
        cv2.imwrite(output, im2)
    # Add seams using the forward cumulative energy function
    elif type_seam_carving == 'stretch' and energy == 'forward':
        for i in range(num_seams):
            e = derivatives(im)
            backtrack = forward_energy_table(e)
            start = find_start(backtrack)
            seam = find_seam(im, backtrack, start)
            seams.append(seam)
            new = remove_seam(im, seam)
            im = new
        for i in range(len(seams)):
            new, adjust_seams = enlarge(im2, seams[i], seams)
            seams = adjust_seams
            im2 = new
        cv2.imwrite(output, im2)
    else:
        print("Error, please check your input")