import os 
import time
import itertools
import numpy as np
import pandas as pd

base_directory = os.getcwd()

experiments_folder = base_directory +'/'+time.strftime("%Y:%m:%d--%H:%M:%S")

print base_directory
print experiments_folder

if not os.path.exists(experiments_folder):
    os.makedirs(experiments_folder)

    velocities = [0.0, 0.5, 0.0, 0.0, -0.5, 0.0]

    positions = [(0.0, 0.5, 0.0, 0.0, -0.5, 0.0), \
                 (0.1, 0.5, 0.0, 0.0, -0.5, 0.0)]

    Walls = [[4, 0, 0, -100.0, 1.0, 0.0, 100.0, 1.0, 0.0], \
             [4, 0, 0, 100.0, 1.0, 0.0, 100.0, -1.0, 0.0], \
             [4, 0, 0, 100.0, -1.0, 0.0, -100.0, -1.0, 0.0], \
             [4, 0, 0, -100.0, -1.0, 0.0, -100.0, 1.0, 0.0]]
    numwalls = 4

    mydata = {}
    mydata['N'] = [2] # number of particles
    mydata['D'] = [2] # Dimension
    mydata['R'] = np.linspace(0.05, 0.25, num=11) # Radii
    mydata['position'] = range(len(positions))
    mydata['mass'] = [2]
    mydata['spin'] = [0.2, 0.4, 0.6]

    all_files = pd.DataFrame(list(itertools.product(*mydata.values())), columns=mydata.keys())

    for counter, row in all_files.iterrows():
        this_experiment = experiments_folder + '/' +str(counter)
        os.makedirs(this_experiment)
        os.chdir(this_experiment)

        parameter_file = open('input_file','w')
        parameter_file.write('# dimension\n')
        parameter_file.write(str(int(row['D'])))
        parameter_file.write('\n# number of gas particles\n')
        parameter_file.write(str(int(row['N'])))
        parameter_file.write('\n# size of gas particles\n')
        parameter_file.write(str(row['R']))
        parameter_file.write('\n# number of collision steps\n')
        parameter_file.write('1000')
        parameter_file.write('\n# number of walls\n')
        parameter_file.write(str(numwalls))
        parameter_file.write('\n# wall type and temp and endpoints\n')
        for i in range(numwalls): 
            newline = ' '.join([str(a) for a in Walls[i]])+'\n'
            parameter_file.write(newline)
        parameter_file.write('# initialize particles? a,g,px,py,pz,r,m,vx,vy,vz\n')
        parameter_file.write('1\n')
        """
        for i in range(int(row['N'])): 
            p = positions[int(row['position'])]
            newline = str(row['spin'])+' 0.7071 '+' '.join([str(p[a]) for a in range(3)])+' '+str(row['R'])+' 2.0 '+' '.join([str(p[a]) for a in range(3, 6)])+'\n'
            parameter_file.write(newline)
        """
        newline = str(row['spin'])+' 0.7071 0.0  0.5 0.0 '+str(row['R'])+' 2.0  0.0 -1.0 0.0\n'
        parameter_file.write(newline)
        newline = str(row['spin'])+' 0.7071 0.0 -0.5 0.0 '+str(row['R'])+' 2.0  0.0  1.0 0.0\n'
        parameter_file.write(newline)
        parameter_file.write('# output directory\n')
        parameter_file.write(str(os.getcwd()) +'/\n')
        parameter_file.close()

        #os.system(base_directory+'/a.out')
        os.system(base_directory + "/a.out input_file")
else: 
    pass

