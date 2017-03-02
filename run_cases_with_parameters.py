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

    velocities = [(0.0, -0.5, 0.0), (0.0, 0.5, 0.0)]
    positions  = [0.5, -0.5]

    Walls = [[4, 0, 0, -100000.0,  1.0, 0.0,  100000.0,  1.0, 0.0], \
             [4, 0, 0,  100000.0,  1.0, 0.0,  100000.0, -1.0, 0.0], \
             [4, 0, 0,  100000.0, -1.0, 0.0, -100000.0, -1.0, 0.0], \
             [4, 0, 0, -100000.0, -1.0, 0.0, -100000.0,  1.0, 0.0]]
    numwalls = 4

    mydata = {}
    mydata['N'] = [2] # number of particles
    mydata['D'] = [2] # Dimension
    mydata['R'] = np.linspace(0.05, 0.25, num=11) # Radii
    mydata['mass'] = [2]
    mydata['spin'] = [0.2, 0.4, 0.6, -0.2, -0.4, -0.6]
    mydata['offsets'] = np.linspace(0.0, 0.5, num=21)

    all_files = pd.DataFrame(list(itertools.product(*mydata.values())), columns=mydata.keys())

    final_value = 0
    offset = 0
    for counter, row in all_files.iterrows():
        final_value = final_value + 1
        this_experiment = experiments_folder + '/' +str(counter)
        os.makedirs(this_experiment)
        os.chdir(this_experiment)

        pos1 = str( row['offsets'])+'  '+str(positions[0])+' 0.0 '
        pos2 = str(-row['offsets'])+' '+str(positions[1])+' 0.0 '
        spin1 = str(row['spin'])
        spin2 = str(abs(row['spin']))
        m = str(row['mass'])
        r = str(row['R'])
        n = str(int(row['N']))

        parameter_file = open('input_file','w')
        parameter_file.write('# dimension\n')
        parameter_file.write(str(int(row['D'])))
        parameter_file.write('\n# number of gas particles\n')
        parameter_file.write(n)
        parameter_file.write('\n# size of gas particles\n')
        parameter_file.write(r)
        parameter_file.write('\n# number of collision steps\n')
        parameter_file.write('100000')
        parameter_file.write('\n# number of walls\n')
        parameter_file.write(str(numwalls))
        parameter_file.write('\n# wall type and temp and endpoints\n')
        for i in range(numwalls): 
            newline = ' '.join([str(a) for a in Walls[i]])+'\n'
            parameter_file.write(newline)
        parameter_file.write('# initialize particles? a,g,px,py,pz,r,m,vx,vy,vz\n')
        parameter_file.write('1\n')

        newline = spin1+' 0.7071 '+pos1+r+' '+m+' 0.0 -1.0 0.0\n'
        parameter_file.write(newline)
        newline = spin2+' 0.7071 '+pos2+r+' '+m+' 0.0  1.0 0.0\n'
        parameter_file.write(newline)
        parameter_file.write('# output directory\n')
        parameter_file.write(str(os.getcwd()) +'/\n')
        parameter_file.close()

        os.system(base_directory + "/a.out input_file")

        extrema = pd.read_csv('output.csv', header=False, delim_whitespace=True)
        extrema.columns=['x1','y1','r1','x2','y2','r2']
        minmax = max(abs(extrema['x1']).max(), abs(extrema['x2']).max())

        result = open('minmax','w')
        result.write(str(minmax))
        result.close()

    """
    for counter, row in all_files.iterrows():
        this_experiment = experiments_folder + '/' +str(counter+final_value)
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

        newline = str(row['spin'])+' 0.7071 0.0  0.5 0.0 '+str(row['R'])+' 2.0  0.0 -1.0 0.0\n'
        parameter_file.write(newline)
        newline = str(-row['spin'])+' 0.7071 0.0 -0.5 0.0 '+str(row['R'])+' 2.0  0.0  1.0 0.0\n'
        parameter_file.write(newline)
        parameter_file.write('# output directory\n')
        parameter_file.write(str(os.getcwd()) +'/\n')
        parameter_file.close()

        os.system(base_directory + "/a.out input_file")

        extrema = pd.read_csv('output.csv', header=False, delim_whitespace=True)
        extrema.columns=['x1','y1','r1','x2','y2','r2']
        minmax = max(abs(extrema['x1']).max(), abs(extrema['x2']).max())

        result = open('minmax','w')
        result.write(str(minmax))
        result.close()
    """

else: 
    pass

