#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "/usr/local/cuda/samples/common/inc/helper_math.h"


/***************************************************************
**      STRUCTURE

The main function is called n_body.  Program flow: main -> control -> n_body

**	TAGS

Particles are indexed 0, 1, 2, ..., N-1
Walls are indexed -1, -2, -3, ...
Particles carry tags that indicate which object(s) were involved in its previous collision event.

At each step, for each particle, we compute time to collision with every particle and wall.
However, we do not want to include times for the objects it hit in the previous step.
These times should be exactly zero, but they could have floating point error and appear to be very small
positive numbers.  This will defeat our logic.  Tags are the fix.

max_complex is the largest number of walls a particle may hit simulateously (see HANDLING COMPLEX COLLISIONS)
So each particle carries a list of max_complex tags.  by default, tags are the particle's own index.
The tags list for particle i is written in slots i*max_complex thru (i+1)*max_complex-1 of tag_CP


**     NOTE FOR FUTURE VERSIONS IF WALLS CURVE

In the current version, a particle carries the same tags until its next collision event.
Thus, we continue not computing time to collision with that object after they have separated.
This is fine if all wall are convex, since a particle can not make consecutive collisions
on the same wall or with the same particle.
But for concave walls, this is not true.  We should rethink this logic.


**    HANDLING COMPLEX COLLISIONS   

The logic of this program is greatly complicated by the possibiliiy of "conplex collisions" involving
multiple particles and/or multiple walls.  These are quite rare, but can occasionally lead to 
catastrophic failure if not handled.  So we take great care to handle such events.  Sadly, it adds
substantial complexity.  We describe that here.

We allow only 2 type of events
Type I: 1 particle hitting >=1 wall (no additional particles)
	simple: 1 particle, 1 wall
	complex (corner): 1 particle, >=2 walls
Type II: 2 particles hitting each other (no additional particles, no walls)

We allow multiple simultaneous collisions of the above types at different locations (see below).

However, we do not allow collisions that are more complex.  There are 2 types.
Type III: >=2 particles and >=1 walls
Type IV: >=3 particles
(a collision that meets both decriptions will be handled as type III)
We will take care to detect more complex collisions and avoid them as follows.

Def: first = particle with smallest index label

Def complex particls = particle involved in a complex collision

Def: dilate = mulitply dt by a number slightly less than 1.  This happpens BEFORE updating position
via new_pos = old_pos + vel * dt.  Dilation results in particles moving slightly
less far than they should.  Therefore expected collisions do not occur during this step. 

Type III fix: for first complex particle, resolve all wall collisions.  Dilate dt for ALL other particles. Turns into type I.
Type IV fix: for first and second complex particles, resolve collision.  Dilate dt for ALL other particles.  Turn into type II.

Note that ALL particles are dilated except the 1 or 2 to be resolved.  This includes particles not
involved in the complex collision.  We had considered only dilating particles involved in the complex
collision, but realized that this could "back" a dilated particle into another undilated particle.


***     HANDLING TYPE I COMPLEX (CORNER) COLLISIONS

Type I simple = 1 particle, 1 wall.
Type I complex (corner) = 1 particle, >=2 walls.

For corner collisions, we resolve each wall collision separately and sequentially.
However, this could result in a trajectory pointed through a wall and out of the billiard cell.
This could happen if some of the walls uses a non-specular reflection law, or if the corner angle is acute.
To fix this, we loop through the walls again and do another SPECULAR reflection at each
wall that particle's trajectory is pointed through.
We may need to repeat this loop mulitple times if the corner is very acutre.


*** HANDLING MULTIPLE SIMULTANEOUS, SPATIALLY SEPARATED EVENTS

Program handles multiple simultaneous events that are spatially separated.
If all of the simultaneous events are of types I or II, then each is resolved independently.
However, if ANY of the events are of types III or IV, the dilation rules above are in force.
This means that only one event will be resolved in that step; the others will be delayed.
They are likely to reoccur in the next step (unless disrupted by the result of the one collision
that was resolved).

************************************************************************************/

//wall types for collision 
#define passive 0
#define heated  1
#define sliding 2

#define MIN(x, y) (x < y) ? x : y
#define MAX(x, y) (x > y) ? x : y

#define hist_length 1000
#define tol_float 0.0001 //measure of "simultaneity" - events within tol_float are simultaneous

std :: random_device generator;
std :: uniform_real_distribution<float> unif_dist(0.0, 1.0);
std :: normal_distribution<float> norm_dist(0.0, 1.0);

dim3 block, grid;

//physical parameters
int DIMENSION = 3;
int N = 10;
float MAX_CUBE_DIM = 2.0;
float surface_area;
float vol;
float default_radius = 0.1;
float default_mass = 2.0;

//non-physical parameters
int MAX_STEPS = 1000;
int steps_per_record = 50;
int track_large_particle = 0;
int ignore_particle_interaction;
int all_particles_diffused = 0;
int diff_number = 2;
int packing_scheme = 0;

//physical constants
float BOLTZ_CONST = 1.0; //1.38064852e-23;
int max_complex = 4;  // number of walls that a particle can hit at a time

//file names for IO
char *in_fname = NULL;
char dir_name[256] = "\0./";
int collides[2];
int collider[2];

//dynamical variables
float3 *p_CPU, *p_GPU; // position
float3 *v_CPU, *v_GPU; // velocity
float *mass_CPU, *mass_GPU; // mass
float *radius_CPU, *radius_GPU; // radius
float *max_square_displacement = NULL; // dist from wall
float *p_temp_CPU; // kinetic energy
float *dt_CPU, *dt_GPU; // time particle will take to collide
float t_tot = 0.0;
float default_p_temp = 1.0; // default kinetic energy
float *p_collisions = NULL;//mean free path
float *w_collisions_heated = NULL;//mean free path
float *w_collisions_passive= NULL;//mean free path
float3 collision_normal;

//walls
float3 normal[6];
float3 tangent1[6];
float3 tangent2[6];
float alpha[6];
int wall_type[6];
float wall_temp[6][2];
float max_temp = 90.0;
float min_temp = 45.0;

//memory management so particle collisions happen in order
int *tag_CPU, *tag_GPU;
int *what_w_CPU, *what_w_GPU;
int *what_p_CPU, *what_p_GPU;
int *how_many_p_CPU, *how_many_p_GPU;
int *how_many_w_CPU, *how_many_w_GPU;
int complex_colliders;
int * complex_event_particle;
int * complex_event_log;




//Trackx thermodynamical quantities.  We want to maintain a record of the thermodynamical state of system
//over the past "window" of events.  This class helps efficiently compute mean and sd for the current event.
class Thermo_Record {
	private:
		float *X;  // array of most recent values of thermo
		float *X2;  // square of X
		float SX;  // sum of values in X
		float SX2;  // sum of values in SX
		int idx;  // points to the oldest array position. We will overwrite this at the next event.
		int window_size;  // length of arrays X, X2
    

	public:
		float window_mean;  // SX / length(SX)
		float window_sd;  // SX2 / length(SX2) - mean^2
		float latest_value;  //last thing put in 

		Thermo_Record(int n)
		{
			window_size = n;
			SX = SX2 = 0.0;
			idx = 0;
			window_mean = window_sd = 0.0;
			latest_value = 0.0;
			X = (float*)calloc(window_size, sizeof(float));
			X2 = (float*)calloc(window_size, sizeof(float));
		}

		~Thermo_Record()
		{
			SX = SX2 = window_mean = window_sd = 0.0;
			if(X != NULL) { free(X); }
			if(X != NULL) { free(X2); }
		}

		void update(float value)
		{
			SX -= X[idx];
			X[idx] = value;
			SX += value;

			window_mean = SX / window_size;

			SX2 -= X2[idx];
			X2[idx] = value * value;
			SX2 += X2[idx];

			window_sd = SX2 / window_size - window_mean * window_mean;
			idx = ( (idx + 1 ) >= window_size) ? 0 : idx + 1;

			latest_value = value;
		}
};


float impulse_sum = 0.0;
float heat_sum = 0.0;
float entropy_sum = 0.0;
int thermo_rec_length = 1000;
Thermo_Record pressure = Thermo_Record(thermo_rec_length);
Thermo_Record gas_temperature = Thermo_Record(thermo_rec_length);

void compute_thermodynamics(float3 v_in, float3 v_out, float3 normal, float wall_temp, float mass, float t_tot)
{
	float impulse, P;
	impulse = mass * ( dot(v_out - v_in, normal));
	impulse_sum += impulse;
	P = (impulse_sum / t_tot) / surface_area;
	pressure.update(P);

	float gas_temp = 0.0;
	for(int i = 0; i < N; i++)
	{
		gas_temp += mass_CPU[i] * dot(v_CPU[i], v_CPU[i]) 
				/ (1.0 * DIMENSION * BOLTZ_CONST);
	}
	gas_temp /= (1.0 * N);
	gas_temperature.update(gas_temp);

	float q_in = mass * dot(v_in, v_in);
	float q_out = mass * dot(v_out, v_out);
	float dq = q_out - q_in;

	heat_sum += dq;
	entropy_sum +=dq/wall_temp;
}




// reads input file of parameters (temperature of walls, particle size, &c)
void read_input_file()
{
  FILE * fp = NULL;
	const int bdim = 132;
	char buff[bdim];
	int i, d;
	double f, g;
	char s[256];

	//Default values
	//physical parmaeters
	DIMENSION = 3;
	N = 25;
	MAX_CUBE_DIM = 4.0;
	default_radius = 0.1;
	default_mass = 2.0;

	//non-physical parameters
	MAX_STEPS = 1000;
	steps_per_record = 50;
	track_large_particle = false;
	ignore_particle_interaction = false;

	//walls
	wall_type[0] = wall_type[1] = heated;
	wall_type[2] = wall_type[3] = wall_type[4] = wall_type[5] = passive;
	wall_temp[0][0] = wall_temp[0][1] = 45.0;
	wall_temp[1][0] = wall_temp[1][1] = 90.0;
	wall_temp[2][0] = wall_temp[2][1] = wall_temp[3][0] = wall_temp[3][1] = 0.0;
	wall_temp[4][0] = wall_temp[4][1] = wall_temp[5][0] = wall_temp[5][1] = 0.0;

	if( (fp = fopen(in_fname,"r")) == NULL)
	{
		printf("No input file. Using default values.\n");
	}
	else
	{
		fgets(buff,bdim,fp);
		fgets(buff,bdim,fp);
		sscanf(buff, "%d", &d);
		DIMENSION = d;

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%d", &d);
		N = d;

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%lf", &f);
		default_radius = f;
		if(default_radius > 0)
		{
		}
		else
		{
			ignore_particle_interaction = true;
		}

		fgets(buff,bdim,fp);
		fgets(buff,bdim,fp);
		sscanf(buff, "%d", &d);
		packing_scheme = d;

		fgets(buff,bdim,fp);
		fgets(buff,bdim,fp);
		sscanf(buff, "%d", &d);
		MAX_STEPS = d;

		fgets(buff, bdim, fp);
		for(i = 0; i < 6; i++)
		{
			fgets(buff, bdim, fp);
			sscanf(buff, "%d %lf %lf", &d, &f, &g);
			wall_type[i] = d;
			wall_temp[i][0] = f;
			wall_temp[i][1] = g;
		}
		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff,"%d", &d);
		track_large_particle = d;

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%s", s);
		strcpy(dir_name, s);

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%d", &d);
		steps_per_record = d;
		fclose(fp);
	}
}


//initialize particles with position, velcoity, radius, etc. 
void pack_particles()
{
	int i, j, k, num, particles_per_side;
	float max_radius, hole_radius, temp, rad_sep = 1.0 + 1.0 / 20.0;
	float T;
	float x_length, x_start;
	float y_length, y_start;
	float z_length, z_start;

	//tag particles not to hit themselves
	for(i = 0;  i < max_complex * N; i++) tag_CPU[i] = -N;//i / max_complex;
	
	//set initial particle parameters
	for (i = 0; i < N; i++)
	{
		mass_CPU[i] = default_mass;
		radius_CPU[i] = default_radius;
		p_temp_CPU[i] = default_p_temp;
	}

	if (track_large_particle)
	{
		radius_CPU[0] = 3.0 * radius_CPU[0];
		mass_CPU[0] = 3.0 * mass_CPU[0];
	}
	max_radius = radius_CPU[0];

	if(packing_scheme < 1)
	{
		//square lattice
		hole_radius = max_radius * rad_sep;
		temp = pow((float)N, 1.0 / (1.0 * DIMENSION)) + .99999;
		particles_per_side = (int)temp;
		x_length = hole_radius * (2.0 * particles_per_side);
		y_length = x_length;
		z_length = x_length;
		if (x_length > 2.0 * MAX_CUBE_DIM)
		{
			printf("Box not big enough to hold all the gas particles.");
			exit(1);
		}
		x_start = -x_length / 2.0 + hole_radius;
		y_start = -y_length / 2.0 + hole_radius;
		z_start = -z_length / 2.0 + hole_radius;
		num = 0;
		for (k = 0; k < particles_per_side; k++)
		{
			for (j = 0; j < particles_per_side; j++)
			{
				for (i = 0; i < particles_per_side; i++)
				{
					if (N <= num) break;
					p_CPU[num].x = x_start + 2.0*hole_radius*i;
					p_CPU[num].y = y_start + 2.0*hole_radius*j;
					p_CPU[num].z = z_start + 2.0*hole_radius*k;
					num++;
				}
			}
		}
	}
	else
	{
		//tetrahedral lattice
		hole_radius = max_radius * rad_sep;
		temp = pow((float)N, 1.0 / (1.0 * DIMENSION)) + .99999;
		particles_per_side = (int)temp;
		x_length = hole_radius * (2.0 * particles_per_side + 1.0);
		y_length = x_length;
		z_length = hole_radius * (sqrt(2) * (particles_per_side - 1) + 2.0);
		if (x_length > MAX_CUBE_DIM)
		{
			printf("Box not big enough to hold all the gas particles.");
			exit(1);
		}
		x_start = -x_length / 2.0 + hole_radius;
		y_start = -y_length / 2.0 + hole_radius;
		z_start = -z_length / 2.0 + hole_radius;
		num = 0;
		for (k = 0; k < particles_per_side; k++)
		{
			for (j = 0; j < particles_per_side; j++)
			{
				for (i = 0; i < particles_per_side; i++)
				{
					if (N <= num) break;
					p_CPU[num].x = x_start + 2.0*hole_radius*i + (k % 2)*hole_radius;
					p_CPU[num].y = y_start + 2.0*hole_radius*j + (k % 2)*hole_radius;
					p_CPU[num].z = z_start + sqrt(2.0)*hole_radius*k;
					num++;
				}
			}
		}
	}

	if(DIMENSION < 3)
	{
		for(i = 0; i < N; i++)
		{
			p_CPU[i].z = v_CPU[i].z = 0.0;
		}
	}

	for (i = 0; i < N; i++)
	{
		max_square_displacement[i] = ((MAX_CUBE_DIM - radius_CPU[i]) + tol_float) * ((MAX_CUBE_DIM - radius_CPU[i]) + tol_float);
	}

	for (i = 0; i < N; i++)
	{
		T = sqrt(BOLTZ_CONST * p_temp_CPU[i] / mass_CPU[i]);

		v_CPU[i].x = norm_dist(generator)*T;
		v_CPU[i].y = norm_dist(generator)*T;
		v_CPU[i].z = norm_dist(generator)*T;
		p_collisions[i] = w_collisions_heated[i] = w_collisions_passive[i] = 0.0;
	}
}


void make_orthonormal_frame(float3 * n, float3 * t1, float3 * t2)
{

	if( ( (*n).x * (*n).x > 0) || ((*n).y * (*n).y > 0) )
	{
		(*t1).x = (*n).y;
		(*t1).y =-(*n).x;
		(*t1).z = 0;
	}
	else if((*n).z * (*n).z > 0)
	{
		(*t1).x =-(*n).z;
		(*t1).y = 0;
		(*t1).z = (*n).x;
	}
	else
	{
		printf("Failed to initialize normal and tangent vectors");
		exit(1);
	}
	(*t2) = cross((*n), (*t1));
	(*n) = normalize(*n);
	(*t1) = normalize(*t1);
	(*t2) = normalize(*t2);
}		


void set_initial_conditions()
{
	int i;

	//CPU MEMORY ALLOCATION
	p_CPU			= (float3*)malloc(		N * sizeof(float3) );
	v_CPU			= (float3*)malloc(		N * sizeof(float3) );
	radius_CPU		= (float* )malloc(		N * sizeof(float ) );
	max_square_displacement = (float* )malloc(		N * sizeof(float ) );
	mass_CPU		= (float* )malloc(		N * sizeof(float ) );
	dt_CPU			= (float* )malloc(		N * sizeof(float ) );
	p_temp_CPU		= (float* )malloc(		N * sizeof(float ) );
	tag_CPU			= (int*   )malloc(max_complex *	N * sizeof(int   ) );
	how_many_p_CPU 		= (int*   )malloc(		N * sizeof(int   ) );
	how_many_w_CPU 		= (int*   )malloc(		N * sizeof(int   ) );
	what_p_CPU		= (int*   )malloc(max_complex *	N * sizeof(int   ) );
	what_w_CPU		= (int*   )malloc(max_complex *	N * sizeof(int   ) );
	p_collisions		= (float* )malloc(		N * sizeof(float ) );
	w_collisions_heated	= (float* )malloc(		N * sizeof(float ) );
	w_collisions_passive	= (float* )malloc(		N * sizeof(float ) );
	complex_event_particle 	= (int*   )malloc(		N * sizeof(int   ) );
	complex_event_log 	= (int*   )malloc(		N * sizeof(int   ) );

	// GPU MEMORY ALLOCATION
	block.x = 512;
	block.y = 1;
	block.z = 1;

	grid.x = (N - 1) / block.x + 1;
	grid.y = 1;
	grid.z = 1;

	cudaMalloc( (void**)&p_GPU,       N *sizeof(float3) );
	cudaMalloc( (void**)&v_GPU,       N *sizeof(float3) );
	cudaMalloc( (void**)&radius_GPU,  N *sizeof(float ) );
	cudaMalloc( (void**)&mass_GPU,    N *sizeof(float ) );

	cudaMalloc( (void**)&tag_GPU,	  max_complex * N *sizeof(int  ) );
	cudaMalloc( (void**)&dt_GPU,			N *sizeof(float) );
	cudaMalloc( (void**)&how_many_p_GPU,		N *sizeof(int  ) );
	cudaMalloc( (void**)&how_many_w_GPU,		N *sizeof(int  ) );
	cudaMalloc( (void**)&what_p_GPU,  max_complex * N *sizeof(int  ) );
	cudaMalloc( (void**)&what_w_GPU,  max_complex * N *sizeof(int  ) );


	// set up wall parameters for box
	surface_area = 6*(2*MAX_CUBE_DIM)*(2*MAX_CUBE_DIM);
	vol = (2*MAX_CUBE_DIM)*(2*MAX_CUBE_DIM)*(2*MAX_CUBE_DIM);
	max_temp = min_temp = 0.0; // average temperature of walls
	for(i = 0; i < 6; i++)
	{
		max_temp = MAX(max_temp, wall_temp[i][0]);
		min_temp = MIN(min_temp, wall_temp[i][0]);
	}
	default_p_temp = (max_temp + min_temp) / 2.0;

	//set entropy and collisions to 0
	for (i = 0; i < 6; i++){ alpha[i] = 1.0;}

	//normal vectors to walls 
	normal[0].x = 1.0; normal[0].y = 0.0; normal[0].z = 0.0;
	normal[2].x = 0.0; normal[2].y = 1.0; normal[2].z = 0.0;
	normal[4].x = 0.0; normal[4].y = 0.0; normal[4].z = 1.0;
	normal[1].x =-1.0; normal[1].y = 0.0; normal[1].z = 0.0;
	normal[3].x = 0.0; normal[3].y =-1.0; normal[3].z = 0.0;
	normal[5].x = 0.0; normal[5].y = 0.0; normal[5].z =-1.0;

	for(i = 0; i < 6; i++)
	{
		make_orthonormal_frame(&normal[i], &tangent1[i], &tangent2[i]);
	}

	// set up particle parameters
	pack_particles();
	//compute_t();

	// copy CPU initialization to GPU
	cudaMemcpy( p_GPU,      p_CPU,                  N *sizeof(float3), cudaMemcpyHostToDevice );
	cudaMemcpy( v_GPU,      v_CPU,                  N *sizeof(float3), cudaMemcpyHostToDevice );
	cudaMemcpy( mass_GPU,   mass_CPU,               N *sizeof(float ), cudaMemcpyHostToDevice );
	cudaMemcpy( radius_GPU, radius_CPU,             N *sizeof(float ), cudaMemcpyHostToDevice );
	cudaMemcpy( tag_GPU,    tag_CPU,  max_complex * N *sizeof(int   ), cudaMemcpyHostToDevice );

}



// evaluates temperature of wall at a point assuming T varies linearly between ends
float wall_temperature(float x, int wall)
{
	return (x + MAX_CUBE_DIM) * (wall_temp[wall][0] + (wall_temp[wall][1] - wall_temp[wall][0]) / (2.0 * MAX_CUBE_DIM));
}



float get_intersection_point(float time, int p)
{
	return p_CPU[p].x + v_CPU[p].x * time;
}


__device__ int particle_wall_collision(float3 * p, float3 * v, float * radius, float max_cube_dim, int i0, int i1, float * t)
{

	int collides = 0;
	float tt, max_cube_minus_radius;
	max_cube_minus_radius = max_cube_dim - radius[i0];
	tt = -2;

	if( (i1 == 0) && (v[i0].x * v[i0].x > 0.0) )
	{
		tt = ( (-max_cube_minus_radius) - p[i0].x ) / v[i0].x;
	}
	else if( (i1 == 2) && (v[i0].y * v[i0].y > 0.0) )
	{
		tt = ( (-max_cube_minus_radius) - p[i0].y ) / v[i0].y;
	}
	else if( (i1 == 4) && v[i0].z * v[i0].z > 0.0)  
	{
		tt = ( (-max_cube_minus_radius) - p[i0].z ) / v[i0].z;
	}
	else if( (i1 == 1) && (v[i0].x * v[i0].x > 0.0) )
	{
		tt = ( max_cube_minus_radius - p[i0].x ) / v[i0].x;
	}
	else if( (i1 == 3) && (v[i0].y * v[i0].y > 0.0) )
	{
		tt = ( max_cube_minus_radius - p[i0].y ) / v[i0].y;
	}
	else if( (i1 == 5) && (v[i0].z * v[i0].z > 0.0) )
	{
		tt = ( max_cube_minus_radius - p[i0].z ) / v[i0].z;
	}
	if( tt >= 0.0)
	{
	  	(*t) = tt;
		collides = 1;
	}

	return collides;
}



__device__ int particle_particle_collision(float3 * p, float3 * v, float * radius, int i0, int i1, float max_cube_dim, float * t)
{
	float xc, yc, zc, xv, yv, zv;
	float discriminant, dd, a, b, c, dt;
	int collides = 0;

	dd = (radius[i0] + radius[i1]) * (radius[i0] + radius[i1]);

	xc = p[i0].x - p[i1].x;
	yc = p[i0].y - p[i1].y;
	zc = p[i0].z - p[i1].z;

	if( (xc * xc + yc * yc + zc * zc) > dd){

		xv = v[i0].x - v[i1].x;
		yv = v[i0].y - v[i1].y;
		zv = v[i0].z - v[i1].z;

		a = xv * xv + yv * yv + zv * zv;
		b = 2.0 * (xc * xv + yc * yv + zc * zv);
		c = (xc * xc) + (yc * yc) + (zc * zc) - dd;

		discriminant = b * b - (4.0 * a * c);

		if(discriminant >= 0.0)
		{
			if(a * a > 0.0) // solve ax^2 + bx + c = 0
			{
				// choose the smallest positive root
				dt = (-b - sqrt(discriminant)) / (2.0 * a);
				if(dt < 0.0)
				{
					dt = (-b + sqrt(discriminant)) / (2.0 * a);
				}
			}
			else if(b * b > 0) // solve bx + c = 0 
			{
				dt = -c / b;
			}
			else if(c * c > 0)
			{
				dt = 0.0;
			}
			else
			{
				dt = -1.;
			}
			if(dt >= 0.0)
			{
				collides = 1;
				*t = dt;
			}
		}
	}
	return collides;
}



__global__ void find_dts(float3 * p, float3 * v, float * radius, float * mass,  // particle data--position, velocity, radius 
			int * tag, int * how_many_p, int * how_many_w, int * what_p_hit, int * what_w_hit, // memory management--what each particle hits and has hit
			int n, int max_complex, float max_cube_dim, int ignore_particle_interaction, float * min_dt) // macros--number of particles, shape of geometry, &c.
{
	float dt, current_min_dt = 0.0;
	int j, k, ok, collides, first_collision, this_particle;

	first_collision = 1;
	current_min_dt = 20000000;

	this_particle = blockDim.x * blockIdx.x + threadIdx.x;

	if(this_particle < n)
	{
		how_many_p[this_particle] = 0;
		how_many_w[this_particle] = 0;

		first_collision = 1;

		if(ignore_particle_interaction)
		{
			// do nothing-only interested in particle-wall collisions
		}
		else
		{
			// check current particle against all particles for collision
			for(j = 0; j < n; j++)			
			{
				ok = 1;
				if((this_particle == j) || (tag[max_complex * this_particle] == j) || (tag[max_complex * j] == this_particle) )ok = 0;
				if(ok > 0)
				{
				  	collides = particle_particle_collision(p, v, radius, this_particle, j, max_cube_dim, &dt);
					if(collides > 0)
					{
						if(first_collision > 0)
						{	
							current_min_dt = dt;
							what_p_hit[max_complex*this_particle] = j;
							first_collision = 0;
							how_many_p[this_particle] = 1;
						}
						else
						{
							if( dt < current_min_dt )
							{
								current_min_dt = dt;
								what_p_hit[max_complex*this_particle] = j;
								how_many_p[this_particle] = 1;
							}
							else if( dt <= current_min_dt)
							{
								current_min_dt = dt;
								what_p_hit[max_complex*this_particle+how_many_p[this_particle]] = j;
								how_many_p[this_particle]++;
							}
						}
					}
				}
			}
		}
		

		// check current particle against walls for collision
		how_many_w[this_particle] = 0;
		for(j = 0; j < 6; j++)
		{
			ok = 1;
			for(k = 0; k < max_complex; k++) 
			{
				if(tag[max_complex * this_particle + k] == -(j + 1) )
				{
					ok = 0;
				}
			}
			if(ok > 0)
			{
				collides = particle_wall_collision(p, v, radius, max_cube_dim, this_particle, j, &dt);
				if( collides > 0 )
				{
					if(first_collision > 0)
					{
						current_min_dt = dt;
						what_w_hit[max_complex * this_particle] = -(j + 1);
						first_collision = 0;
						how_many_w[this_particle] = 1;
						how_many_p[this_particle] = 0;
					}
					else
					{
						if( dt < current_min_dt)
						{
							current_min_dt = dt;
							what_w_hit[max_complex * this_particle] = -(j + 1);
							how_many_w[this_particle] = 1;
							how_many_p[this_particle] = 0;
							
						}
						else if( dt <= current_min_dt)
						{
							what_w_hit[max_complex * this_particle + how_many_w[this_particle]] = -(j + 1);
							how_many_w[this_particle]++;
						}
							
					}
				}
			}
		}
		min_dt[this_particle] = current_min_dt;
	}
}


float3 specular_reflect(float3 v_in, float3 n)
{
	float3 v_out = v_in - (2.0 * dot(n, v_in) * n);
	return(v_out);
}

float chi_sq(float u, float sigma)
{
	return sigma * sqrt(fabs(2.0 * log(1.0 - u)));
}


float3 heated_wall_reflection(float3 v_in, float3 n, float3 t1, float3 t2, float T, float m)
{
	float3 v_out;
	float u, sn, st1, st2;
	float sigma = sqrt(BOLTZ_CONST * T / m);

	u = unif_dist(generator);
	sn = chi_sq(u, sigma);
	st1 = norm_dist(generator)*sigma;
	st2 = norm_dist(generator)*sigma;

	v_out = sn*n + st1*t1 + st2*t2;
	return v_out;
}

void particle_particle_collision(int i1, int i2)
{
	float3 n, v1_n, v1_t, v2_n, v2_t;
	float s1_n, s2_n, m1, m2, M;
	p_collisions[i1] += 1.0;
	p_collisions[i2] += 1.0;

	tag_CPU[max_complex * i1] = i2;
	tag_CPU[max_complex * i2] = i1;

	m1 = mass_CPU[i1];
	m2 = mass_CPU[i2];
	M = m1 + m2;

	// n is the vector from center of i0 to center of i1
	n = normalize(p_CPU[i1] - p_CPU[i2]);

	//--------------------Particle 1-----------------------//
	// get vector component of v_CPU[i0] parallel to n
	s1_n = dot(n, v_CPU[i1]);
	v1_n = s1_n * n;
	// get vector component of v_CPU[i0] perpendicular to n
	v1_t = v_CPU[i1] - v1_n;
	//-----------------------------------------------------//


	//--------------------Particle 2-----------------------//
	// get vector component of v_CPU[i1] parallel to n
	s2_n = dot(n, v_CPU[i2]);
	v2_n = s2_n * n;
	// get vector component of v_CPU[i1] perpendicular to n
	v2_t = v_CPU[i2] - v2_n;
	//-----------------------------------------------------//

	// update velocities
	v_CPU[i1] = v1_t + (((m1-m2)*s1_n + (2*m2 )*s2_n)/M) * n;
	v_CPU[i2] = v2_t + (((2*m1 )*s1_n + (m2-m1)*s2_n)/M) * n;
}


void errorCheck(int num, const char * message)
{
	cudaError_t error;
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("Cuda Error at time %d: %s = %s\n", num, message, cudaGetErrorString(error));
	}
}
void add_recursively_to_complex_event_particles(int p, int ignored_val)
{
	bool not_included_yet = true;
	if(p != ignored_val)
	{
		for(int i = 0; i < complex_colliders; i++)
			if(complex_event_particle[i] == p)
				not_included_yet = false;
		if(not_included_yet)
		{
			complex_event_particle[complex_colliders] = p;
			complex_colliders++;
		}
		for(int i = 0; i < how_many_p_CPU[p]; i++)
			add_recursively_to_complex_event_particles(what_p_CPU[max_complex*p+i], ignored_val);
		complex_event_log[complex_colliders] = how_many_p_CPU[p];
		how_many_p_CPU[p] = how_many_w_CPU[p] = 0;
	}
}

bool detect_complex_collision_events()
{
	bool anything_complex_found = false;
	int i, j, k;

	complex_colliders = 0;
	for(i = 0; i < N; i++)
	{
		if(how_many_p_CPU[i] > 1)
		{
			add_recursively_to_complex_event_particles(i, -2);
			anything_complex_found = true;
		}
		else if( (how_many_p_CPU[i] > 0) & (how_many_w_CPU[i] > 0) )
		{
			for(j = 0; j < how_many_p_CPU[i]; j++)
			{
				k = what_p_CPU[max_complex*i+j];
				add_recursively_to_complex_event_particles(k, i);
			}
			anything_complex_found = true;

		}
	}
	return anything_complex_found;
}

void check_no_particles_escape(int time_step, float total_time)
{
	for (int i = 0; i < N; i++)
	{
		if( (p_CPU[i].x * p_CPU[i].x > max_square_displacement[i]) ||
		    (p_CPU[i].y * p_CPU[i].y > max_square_displacement[i]) ||
		    (p_CPU[i].z * p_CPU[i].z > max_square_displacement[i]))
		{
			printf("\nError in step %6d: particle %2d escaped!!! It's now at position (%f, %f, %f). Last hit %2d %2d %2d\n",
				time_step, i, p_CPU[i].x, p_CPU[i].y, p_CPU[i].z, tag_CPU[max_complex * i] + 1, tag_CPU[max_complex * i + 1] + 1, tag_CPU[max_complex * i + 2] + 1);
		}
	}
}

void randomize_position(int p)
{
	float lower_bound = -MAX_CUBE_DIM + radius_CPU[p] + tol_float;
	float scale = -2 * lower_bound;
	float px, py, pz, dd;
	float3 new_pos;
	bool needs_new_position = true;

	while(needs_new_position)
	{
		needs_new_position = false;
		px = lower_bound + scale * unif_dist(generator);
		py = lower_bound + scale * unif_dist(generator);
		pz = lower_bound + scale * unif_dist(generator);
		new_pos = make_float3(px, py, pz);

		for(int i = 0; i < N; i++)
		{
			dd = dot(p_CPU[i] - new_pos, p_CPU[i] - new_pos);
			if(dd < (radius_CPU[i] + radius_CPU[p]) * (radius_CPU[i] + radius_CPU[p]) )
				needs_new_position = true;
		}
	}
	p_CPU[p] = new_pos;
}

void n_body()
{
	float dt_step = 0.0;

	FILE * out_file;
	FILE * complex_event_log_file;
	char dir[256];

	int i, j, step, w;
	int smart_stop_found = 0;
	int smart_max_steps = MAX_STEPS;

	float3 v_in, v_out;

	bool complex_collisions_occurred = false;

	set_initial_conditions();

	
	//WRITE INITIAL CONDITION TO FILE
	complex_event_log_file = fopen(strcat(strcpy(dir, dir_name), "complex_events_log.txt"), "w");
	out_file = fopen(strcat(strcpy(dir, dir_name), "output.csv"), "w");
	fprintf(out_file, "#box dimension\n box, %lf\n", MAX_CUBE_DIM);
	fprintf(out_file, "#particle radii\n");
	for(i = 0; i < N; i++)
	{
		fprintf(out_file, "r, %d, %lf, %lf\n", i, radius_CPU[i], mass_CPU[i]);
	}
	for(i = 0; i < N; i++)
	{
		fprintf(out_file, "c, %d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf %lf\n", 
					i, 0, t_tot, 
					p_CPU[i].x, p_CPU[i].y, p_CPU[i].z, 
					0.0, 0.0, 0.0,
					v_CPU[i].x, v_CPU[i].y, v_CPU[i].z,
					collision_normal.x, collision_normal.y, collision_normal.z, 
					0.0, 0.0, 0.0, 0.0
			);
	}

	step = 0;
	smart_max_steps = MAX_STEPS;
	while(step++ <= smart_max_steps)
	{
	  	// on GPU - find smallest time step s.t. any particle(s) collide either 
	  	// with each other or a wall and update all particles to that time step
		find_dts<<<grid, block>>>(p_GPU, v_GPU, radius_GPU, mass_GPU, tag_GPU, how_many_p_GPU, how_many_w_GPU, what_p_GPU, what_w_GPU, int(N), max_complex, MAX_CUBE_DIM, ignore_particle_interaction, dt_GPU);
		errorCheck(step, "find_dts");

		//copy minimum time step and index of corresponding colliding element onto CPU 
		cudaMemcpy( how_many_p_CPU, how_many_p_GPU,             N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy( how_many_w_CPU, how_many_w_GPU,             N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy(     what_p_CPU,     what_p_GPU,max_complex* N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy(     what_w_CPU,     what_w_GPU,max_complex* N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy(         dt_CPU,         dt_GPU,             N * sizeof(float), cudaMemcpyDeviceToHost);

		//find global min dt
		dt_step = dt_CPU[0];
		for (i = 1; i < N; i++) if(dt_CPU[i] <= dt_step) dt_step = dt_CPU[i];
		t_tot += dt_step;
		for (i = 1; i < N; i++) if(dt_CPU[i] > dt_step) how_many_p_CPU[i] = how_many_w_CPU[i] = 0;

		// if no collisions were detected, we are done. 
		if(dt_step < 0.0)
		{
			printf("\nEarly exit : dt_step = %f < 0 at step %i\n", dt_step, step);
			exit(1);
		}

		// check (and modify tags) if complex collision events occurred
		complex_collisions_occurred = detect_complex_collision_events();

		// update all particles to new time step
		for(i = 0; i < N; i++)
		{
			// update particle's position
			p_CPU[i] += v_CPU[i] * dt_step;

			// check if it is involved in a collision with either particle or wall and update velocity accordingly
			if( how_many_w_CPU[i] > 0)
			{
				v_in = v_CPU[i];
				for(j = 0; j < how_many_w_CPU[i]; j++)
				{
					w = -(1 + what_w_CPU[max_complex * i + j]);
					if(wall_type[w] == heated)
					{
						v_out = heated_wall_reflection(v_in, normal[w], tangent1[w], tangent2[w], wall_temp[w][0], mass_CPU[w]);
					}
					else if(wall_type[w] == passive)
					{
						v_out = specular_reflect(v_in, normal[w]);
					}
					else
					{
						printf("Illegal wall tag");
						exit(1);
					}
					tag_CPU[i * max_complex + j] = what_w_CPU[max_complex * i + j];
					v_in = v_out;
				}
				v_CPU[i] = v_out;
			}
			else if( how_many_p_CPU[i] > 0)
			{
				j = what_p_CPU[max_complex * i];
				if(i > j) particle_particle_collision(i, j);
			}
		}

		if( complex_collisions_occurred ) 
		{
			for(i = 0; i < complex_colliders; i++)
			{
				randomize_position(complex_event_particle[i]);
				fprintf(complex_event_log_file, "%d, ", complex_event_log[i]);
			}
			fprintf(complex_event_log_file, "\n");
		}
		//compute smart_stop_condition
		if(smart_stop_found == 0)
		{
			//if(abs(pressure.window_sd / pressure.window_mean) < 0.00001)
			if(t_tot > 20)
			{
				smart_stop_found = 1;
				smart_max_steps = step;
			}
		}	

		
		// update position on GPU to new time step
		// update velocity on GPU to match CPU 
		// (and also tag which keeps track of most recent collision for each particle)
		cudaMemcpy(   p_GPU,   p_CPU,			N * sizeof(float3), cudaMemcpyHostToDevice );
		cudaMemcpy(   v_GPU,   v_CPU,			N * sizeof(float3), cudaMemcpyHostToDevice );
		cudaMemcpy( tag_GPU, tag_CPU, max_complex *	N * sizeof(int   ), cudaMemcpyHostToDevice );

		//fireproofing: check at each time step that no particles escaped.
		check_no_particles_escape(step, t_tot);

		//end of this step
	}
	
	
	/*/  WRITE FINAL CONDITIONS TO FILE /*/
	for(i = 0; i < N; i++)
	{
		fprintf(out_file, "c, %d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", 
					i, 0, t_tot, 
					p_CPU[i].x, p_CPU[i].y, p_CPU[i].z, 
					0.0, 0.0, 0.0,
					v_CPU[i].x, v_CPU[i].y, v_CPU[i].z,
					collision_normal.x, collision_normal.y, collision_normal.z, 
					pressure.latest_value, gas_temperature.latest_value, heat_sum, entropy_sum // dummy for pressure
			);
	}
	fclose(out_file);
	fclose(complex_event_log_file);
	printf("%i gas particles, %d steps, %.4f seconds in time\n", N, step, t_tot);
}



void control()
{	
	clock_t time_0, time_1;
	FILE *fp;
	char dir[256];

	time_0 = clock();
	read_input_file();
    	n_body();
	time_1 = clock();

	printf("\n Runtime %.5f seconds\n", (float)(time_1 - time_0) / CLOCKS_PER_SEC);
	printf("\n DONE \n");

	fp = fopen(strcat(strcpy(dir, dir_name), "log"), "w");
	fprintf(fp, "N, Nsteps, physical_time, runtime, end_pressure, end_entropy_rate\n%d, %d, %lf, %lf, %lf, %lf", N, MAX_STEPS, t_tot, (float)(time_1 - time_0) / CLOCKS_PER_SEC, pressure.latest_value, entropy_sum / t_tot); 
	fclose(fp);
}


int main(int argc, char** argv)
{
	if(--argc < 1)
	{
		printf("Without input file, reverting to default parameters\n");
	}
	else
	{
		in_fname = argv[1];
	}

	control();
	return 0;
}
