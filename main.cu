#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "/usr/local/cuda/samples/common/inc/helper_math.h"

/***************************************************************
**      STRUCTURE

The main function is called n_body.  Program flow: main -> n_body

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

The logic of this program is greatly complicated by the possibiliiy of "complex collisions" involving
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
jype IV: >=3 particles
(a collision that meets both decriptions will be handled as type III)
We will take care to detect more complex collisions and avoid them as follows.

Def: complex particle = particle involved in a complex collision

Def: randomize = relocate particle randomly in domain so that it is not in contact with any other particle. 

Type III fix: randomize all complex particles
Type IV fix: randomize all complex particles

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

************************************************************************************/


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

//wall types for collision 
#define passive 0
#define heated  1
#define sliding 2
#define circles 3
#define no_slip 4
#define end 5

#define MIN(x, y) (x < y) ? x : y
#define MAX(x, y) (x > y) ? x : y

#define timetol 0.01
#define hist_length 1000

std :: random_device generator;
std :: uniform_real_distribution<float> unif_dist(0.0, 1.0);
std :: normal_distribution<float> norm_dist(0.0, 1.0);

dim3 block, grid;

//physical parameters
int DIMENSION = 3;
int N = 10;
float MAX_CUBE_DIM = 4.0;
float MIN_CUBE_DIM =-4.0;
float surface_area;
float vol;
float default_radius = 0.5;
float default_mass = 2.0;

//non-physical parameters
bool no_input_file = true;
int MAX_STEPS = 1000;
int FILE_LINES = 1000;
int track_large_particle = 0;
int ignore_particle_interaction = 0;
bool no_slip_particle = true;
int all_particles_diffused = 0;
int diff_number = 2;

//physical constants
float BOLTZ_CONST = 1.0; //1.38064852e-23;
int max_complex = 4;  // number of walls that a particle can hit at a time

//file names for IO
char *in_fname = NULL;
char dir_name[256] = "\0./";

//dynamical variables
float3 *p_CPU = NULL, *p_GPU = NULL; // position
float3 *v_CPU = NULL, *v_GPU = NULL; // velocity
float *angular_momentum = NULL;
float *no_s_gamma = NULL;
float *mass_CPU = NULL, *mass_GPU = NULL; // mass
float *radius_CPU = NULL, *radius_GPU = NULL; // radius
float *p_temp_CPU = NULL; // kinetic energy
float *dt_CPU = NULL, *dt_GPU = NULL; // time particle will take to collide
float t_tot = 0.0;
float default_p_temp = 1.0; // default kinetic energy
float3 collision_normal;

//walls
int num_walls = 0;
float max_temp = 90.0;
float min_temp = 45.0;

//memory management so particle collisions happen in order
int *tag_CPU = NULL, *tag_GPU = NULL;
int *what_w_CPU = NULL, *what_w_GPU = NULL;
int *what_p_CPU = NULL, *what_p_GPU = NULL;
int *how_many_p_CPU = NULL, *how_many_p_GPU = NULL;
int *how_many_w_CPU = NULL, *how_many_w_GPU = NULL;
int complex_colliders = 0;
int *complex_event_particle = NULL;
int *complex_event_log = NULL;
float dt_step = 0.0;


class Wall {
	private: 
		int D = DIMENSION; // for collision methods on GPU
		float max_cube_dim = MAX_CUBE_DIM;
	public: 
		int type; // heated, passive, sliding, or circles
		float2 endpoints[2]; // if a 2D line segment
		float3 position; // center if a circle type wall or position if plane in 3d
		float  radius; // radius if a circle type wall
		float3 normal; // normal vector (assume not a circle at the moment)
		float3 tangent1;
		float3 tangent2;
		float wall_temp[2]; // temperature 
		float alpha; // probability a wall is thermally active
	Wall()
	{
		alpha = 1.0;
		D = DIMENSION;
	}

	// returns 0 or 1 whether a particle (p) collides with the wall (0=no collision)
	// and stores the time until collision in float &time parameter. 
	CUDA_CALLABLE_MEMBER int time_to_collision(float3 p, float3 v, float r, float * time)
	{
		int collides = 0;
		if(type != circles) 
		{
			if(D > 2)
			{
				float tt, max_cube_minus_radius;
				max_cube_minus_radius = max_cube_dim - r;
				tt = -2;
				if( (normal.x > 0.5) && (v.x * v.x > 0.0) )
				{
					tt = ( (-max_cube_minus_radius) - p.x ) / v.x;
				}
				else if( (normal.y > 0.5) && (v.y * v.y > 0.0) )
				{
					tt = ( (-max_cube_minus_radius) - p.y ) / v.y;
				}
				else if( (normal.z > 0.5) && v.z * v.z > 0.0)  
				{
					tt = ( (-max_cube_minus_radius) - p.z ) / v.z;
				}
				else if( (normal.x < -0.5) && (v.x * v.x > 0.0) )
				{
					tt = ( max_cube_minus_radius - p.x ) / v.x;
				}
				else if( (normal.y < -0.5) && (v.y * v.y > 0.0) )
				{
					tt = ( max_cube_minus_radius - p.y ) / v.y;
				}
				else if( (normal.z < -0.5) && (v.z * v.z > 0.0) )
				{
					tt = ( max_cube_minus_radius - p.z ) / v.z;
				}
				if( tt >= 0.0)
				{
				  	(*time) = tt;
					collides = 1;
				}
			}
			else
			{
				float B1,B2,A11,A12,A21,A22,D,t1,t2;
				float x1,y1,x2,y2;

				x1 = endpoints[0].x + (r*normal.x);
				y1 = endpoints[0].y + (r*normal.y);
				x2 = endpoints[1].x + (r*normal.x);
				y2 = endpoints[1].y + (r*normal.y);

				B1 = p.x - x1;
				B2 = p.y - y1;

				A11 = (x2 - x1);
				A12 = -v.x;
				A21 = (y2 - y1);
				A22 = -v.y;

				D = A11 * A22 - (A21 * A12);

				if( abs(D) > 0 )
				{
					t1 = (A22*B1 - A12*B2) / D;
					t2 = (A11*B2 - A21*B1) / D;
					// if they cross at a point in the range of the line segment
					if( (t1 >= 0.0) && (t1 <= 1.0) && (t2 >= 0.0))
					{
						collides = 1;
						(*time) = t2;
					}
				}
			}
		}
		else
		{
			// check circle - vector collision (see GPU code)
		}
		return collides;

	}

	// resolves outgoing particle velocity after collision
	float3 resolve_collision(int p, float3 v)
	{
		float3 v_out;
		if(type == heated)
		{
			float sigma, u, sn, st1, st2, T, m;

			if(DIMENSION > 2)
			{
				T = wall_temp[0];
				m = mass_CPU[p];
				sigma = sqrt(BOLTZ_CONST * T / m);

				u = unif_dist(generator);
				sn = sigma * sqrt(fabs(2.0 * log(1.0 - u)));
				st1 = norm_dist(generator)*sigma;
				st2 = norm_dist(generator)*sigma;
				v_out = sn*normal + st1*tangent1 + st2*tangent2;
			}
			else
			{
				T = wall_temp[0];
				m = mass_CPU[p];
				sigma = sqrt(BOLTZ_CONST*T/m);
				u = unif_dist(generator);
				sn = sigma * sqrt(fabs(2.0*log(1.0-u)));
				st1 = norm_dist(generator)*sigma;

				v_out.x = sn*normal.x + st1 * tangent1.x;
				v_out.y = sn*normal.y + st1 * tangent1.y;
				v_out.z = 0.0;
			}
		}
		else if(type == passive)
		{
			v_out = v - (2.0 * dot(normal, v) * normal);
		}
		else if(type == circles)
		{

		}
		else if(type == no_slip)
		{
			float vn,vt,vn0,vt0,a, g;

			g = no_s_gamma[p]; 
			a = angular_momentum[p]; 

			vn = dot(v_CPU[p], normal);
			vt = dot(v_CPU[p], tangent1);

			angular_momentum[p] 	= -((1.0-g*g)/(1.0+g*g))* a + 
						  -(2.0*g / (1.0+g*g)) 	* vt;

			vt0 = (-2.0*g/(1.0+g*g))  * a + 
			      ((1.0-g*g)/(1.0+g*g)) * vt;

			vn0 = -1.0 * vn;

			v_out = vt0 * tangent1 + vn0 * normal;
		}
		return v_out;
	}
};
Wall * walls_CPU = NULL;
Wall * walls_GPU = NULL;

//Tracks thermodynamical quantities.  We want to maintain a record of the thermodynamical state of system
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



void allocate_wall_memory()
{
	walls_CPU = (Wall*)malloc(num_walls * sizeof(Wall));
	for(int i = 0; i < num_walls; i++) walls_CPU[i] = Wall();
}

void setup_walls()
{
	int i;

	surface_area = 6*(2*MAX_CUBE_DIM)*(2*MAX_CUBE_DIM);
	vol = (2*MAX_CUBE_DIM)*(2*MAX_CUBE_DIM)*(2*MAX_CUBE_DIM);

	for(i = 0; i < 6; i++)
	{
		max_temp = MAX(max_temp, walls_CPU[i].wall_temp[0]);
		min_temp = MIN(min_temp, walls_CPU[i].wall_temp[0]);
	}
	default_p_temp = (max_temp + min_temp) / 2.0;

}

// returns true if particle is inside domain, false if outside
bool inside_domain(float3 p, float r)
{
	int num_total;

	if(DIMENSION > 2)
	{
		r*=1.5;
		if( 	(p.x*p.x > ((MAX_CUBE_DIM-r)*(MAX_CUBE_DIM-r))) || 
			(p.y*p.y > ((MAX_CUBE_DIM-r)*(MAX_CUBE_DIM-r))) || 
			(p.z*p.z > ((MAX_CUBE_DIM-r)*(MAX_CUBE_DIM-r))) )
		{
			num_total = 2;
		}
		else
		{
			num_total = 1;
		}
	}
	else
	{
		num_total = 0;
		float t, d = (DIMENSION > 2) ? 1.0 : 0.0;
		float3 v = make_float3(2, 2, 2*d);

		for(int w = 0; w < num_walls; w++)
		{
			num_total += walls_CPU[w].time_to_collision(p, v, r, &t);
		}
	}
	return (num_total%2);
}

// find position chosen randomly in box for particle such that 
// (1) the new position is within the domain
// (2) particle at new position is not in contact / overlapping with any other particles. 
void randomize_position(int p, float MIN_EXTENT, float MAX_EXTENT)
{
	float px, py, pz, dd;
	float3 new_pos;
	bool needs_new_position = true;

	while(needs_new_position)
	{
		needs_new_position = false;

		// create new x, y, z coordinate for particle p
		px = MIN_EXTENT + (MAX_EXTENT-MIN_EXTENT) * unif_dist(generator);
		py = MIN_EXTENT + (MAX_EXTENT-MIN_EXTENT) * unif_dist(generator);
		pz = 0.0;
		if(DIMENSION > 2) 
		{
			pz = MIN_EXTENT + (MAX_EXTENT - MIN_EXTENT) * unif_dist(generator);
		}
		new_pos = make_float3(px, py, pz);

		// check if it is inside domain AND doesn't overlap with any other particles. 
		if( inside_domain(new_pos, radius_CPU[p]) )
		{
			for(int i = 0; i < N; i++)
			{
				dd = dot(p_CPU[i] - new_pos, p_CPU[i] - new_pos);
				if(dd < (radius_CPU[i] + radius_CPU[p]) * (radius_CPU[i] + radius_CPU[p]) )
				{
					needs_new_position = true;
					i = N;
				}
			}
		}
		else
		{
			needs_new_position = true;
		}
	}
	p_CPU[p] = new_pos;
	printf("Looking for positions for particle %d - %lf %lf %lf\n", p,p_CPU[p].x,p_CPU[p].y,p_CPU[p].z);
}


void distribute_particles(float MIN_EXTENT, float MAX_EXTENT, int DIM)
{
	float T, d = (DIM > 2) ? 1.0 : 0.0;
	int i;

	//set initial particle parameters
	for (i = 0; i < N; i++)
	{
		mass_CPU[i] = default_mass;
		radius_CPU[i] = default_radius;
		p_temp_CPU[i] = default_p_temp;
		no_s_gamma[i] = 1.0/sqrt(2.0);
		angular_momentum[i] = 0.0;
	}

	if (track_large_particle)
	{
		radius_CPU[0] = 3.0 * radius_CPU[0];
		mass_CPU[0] = 3.0 * mass_CPU[0];
	}

	for (i = 0; i < N; i++)
	{
		T = sqrt(BOLTZ_CONST * p_temp_CPU[i] / mass_CPU[i]);
		v_CPU[i].x = norm_dist(generator)*T;
		v_CPU[i].y = norm_dist(generator)*T;
		if(DIMENSION > 2) v_CPU[i].z = norm_dist(generator)*T;
		else v_CPU[i].z = 0.0;
	}

	for(i = 0; i < N; i++) p_CPU[i] = 2.0 * MAX_EXTENT * make_float3(1.0, 1.0, d);
	for(i = 0; i < N; i++) randomize_position(i, MIN_EXTENT, MAX_EXTENT);
}

void allocate_CPU_memory()
{
	p_CPU			= (float3*)malloc(		N * sizeof(float3) );
	v_CPU			= (float3*)malloc(		N * sizeof(float3) );
	angular_momentum 	= (float* )malloc(      	N * sizeof(float ) );
	no_s_gamma          	= (float* )malloc(		N * sizeof(float ) );
	radius_CPU		= (float* )malloc(		N * sizeof(float ) );
	mass_CPU		= (float* )malloc(		N * sizeof(float ) );
	dt_CPU			= (float* )malloc(		N * sizeof(float ) );
	p_temp_CPU		= (float* )malloc(		N * sizeof(float ) );
	tag_CPU			= (int*   )malloc(max_complex *	N * sizeof(int   ) );
	how_many_p_CPU 		= (int*   )malloc(		N * sizeof(int   ) );
	how_many_w_CPU 		= (int*   )malloc(		N * sizeof(int   ) );
	what_p_CPU		= (int*   )malloc(max_complex *	N * sizeof(int   ) );
	what_w_CPU		= (int*   )malloc(max_complex *	N * sizeof(int   ) );
	complex_event_particle 	= (int*   )malloc(		N * sizeof(int   ) );
	complex_event_log 	= (int*   )malloc(2 *		N * sizeof(int   ) );

	//tag particles not to hit themselves
	for(int i = 0; i < max_complex * N; i++) tag_CPU[i] = N + 2;
}

void allocate_GPU_memory()
{

	// GPU MEMORY ALLOCATION
	block.x = 1024;
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

	cudaMalloc(&walls_GPU, num_walls*sizeof(Wall));

	// copy CPU initialization to GPU
	cudaMemcpy( p_GPU,      p_CPU,                  N *sizeof(float3), cudaMemcpyHostToDevice );
	cudaMemcpy( v_GPU,      v_CPU,                  N *sizeof(float3), cudaMemcpyHostToDevice );
	cudaMemcpy( mass_GPU,   mass_CPU,               N *sizeof(float ), cudaMemcpyHostToDevice );
	cudaMemcpy( radius_GPU, radius_CPU,             N *sizeof(float ), cudaMemcpyHostToDevice );
	cudaMemcpy( tag_GPU,    tag_CPU,  max_complex * N *sizeof(int   ), cudaMemcpyHostToDevice );

	cudaMemcpy(walls_GPU, walls_CPU, num_walls*sizeof(Wall), cudaMemcpyHostToDevice);
}


// reads input file of parameters (temperature of walls, particle size, &c)
void read_input_file_and_set_initial_conditions()
{
  	FILE * fp = NULL;
	const int bdim = 132;
	char buff[bdim];
	int i, d, dim, n;
	double f, g, h, f1, g1, h1, t1, t2, v1, v2, v3;
	char s[256];
	double MINCD,MAXCD;
	bool particle_initialization_by_file = false;

	if(no_input_file)
	{
		printf("No input file. Using default values.\n");
		num_walls = 6;
		allocate_wall_memory();

		walls_CPU[0].normal.x = 1.0; walls_CPU[0].normal.y = 0.0; walls_CPU[0].normal.z = 0.0;
		walls_CPU[2].normal.x = 0.0; walls_CPU[2].normal.y = 1.0; walls_CPU[2].normal.z = 0.0;
		walls_CPU[4].normal.x = 0.0; walls_CPU[4].normal.y = 0.0; walls_CPU[4].normal.z = 1.0;
		walls_CPU[1].normal.x =-1.0; walls_CPU[1].normal.y = 0.0; walls_CPU[1].normal.z = 0.0;
		walls_CPU[3].normal.x = 0.0; walls_CPU[3].normal.y =-1.0; walls_CPU[3].normal.z = 0.0;
		walls_CPU[5].normal.x = 0.0; walls_CPU[5].normal.y = 0.0; walls_CPU[5].normal.z =-1.0;

		for(i = 0; i < 6; i++)
		{
			walls_CPU[i].type=passive;
			make_orthonormal_frame(&(walls_CPU[i].normal), &(walls_CPU[i].tangent1), &(walls_CPU[i].tangent2));
			walls_CPU[i].position = -MAX_CUBE_DIM * (walls_CPU[i].normal);
		}
		allocate_CPU_memory();
		distribute_particles(MIN_CUBE_DIM, MAX_CUBE_DIM, DIMENSION);
		allocate_GPU_memory();
	}
	else
	{
		fp = fopen(in_fname,"r");

		fgets(buff,bdim,fp);
		fgets(buff,bdim,fp);
		//sscanf(buff, "%d", &dim); DIMENSION = dim;
		sscanf(buff, "%d", &DIMENSION);

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		//sscanf(buff, "%d", &n);
		//N = n;
		sscanf(buff, "%d", &N);
		allocate_CPU_memory();

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		//sscanf(buff, "%lf", &f); default_radius = f;
		sscanf(buff, "%f", &default_radius);
		if(default_radius > 0){ }
		else { ignore_particle_interaction = true; }

		fgets(buff,bdim,fp);
		fgets(buff,bdim,fp);
		//sscanf(buff, "%d", &d); MAX_STEPS = d;
		sscanf(buff, "%d", &MAX_STEPS);

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%d", &num_walls);
		allocate_wall_memory();

		if(DIMENSION < 3) MINCD = MAXCD = 0.0;

		fgets(buff, bdim, fp);
		for(i = 0; i < num_walls; i++)
		{
			fgets(buff, bdim, fp);
			sscanf(buff, "%d %lf %lf %lf %lf %lf %lf %lf %lf",&d,&t1,&t2,&f,&g,&h,&f1,&g1,&h1);

			walls_CPU[i].wall_temp[0] = t1;
			walls_CPU[i].wall_temp[1] = t2;

			walls_CPU[i].type = d;

			if(d != circles)
			{
				if(DIMENSION < 3)
				{
					walls_CPU[i].endpoints[0].x = f;
					walls_CPU[i].endpoints[0].y = g;
					walls_CPU[i].endpoints[1].x = f1;
					walls_CPU[i].endpoints[1].y = g1;
					// inward pointing normals assumes that the walls are given in clockwise orientation. 
					walls_CPU[i].normal = normalize(make_float3((g1-g),(f-f1),0.0));
					walls_CPU[i].tangent1 = normalize(make_float3((f-f1),(g-g1),0.0));
					walls_CPU[i].tangent2 = make_float3(0.0,0.0,0.0);

					MAXCD = MAX(MAXCD, f);
					MAXCD = MAX(MAXCD, g);
					MAXCD = MAX(MAXCD, f1);
					MAXCD = MAX(MAXCD, g1);
					MINCD = MIN(MINCD, f);
					MINCD = MIN(MINCD, g);
					MINCD = MIN(MINCD, f1);
					MINCD = MIN(MINCD, g1);
				}
				else
				{
					walls_CPU[i].position = make_float3(f1,g1,h1);
					walls_CPU[i].normal = normalize(make_float3(f,g,h));
					make_orthonormal_frame(&(walls_CPU[i].normal), 
								&(walls_CPU[i].tangent1), 
								&(walls_CPU[i].tangent2));

					MINCD = MIN(MINCD, f);
					MINCD = MIN(MINCD, g);
					MINCD = MIN(MINCD, h);
					MINCD = MIN(MINCD, f1);
					MINCD = MIN(MINCD, g1);
					MINCD = MIN(MINCD, h1);

					MAXCD = MAX(MAXCD, f);
					MAXCD = MAX(MAXCD, g);
					MAXCD = MAX(MAXCD, h);
					MAXCD = MAX(MAXCD, f1);
					MAXCD = MAX(MAXCD, g1);
					MAXCD = MAX(MAXCD, h1);
				}
			}
			else
			{
				walls_CPU[i].position = make_float3(f,g,h);
				walls_CPU[i].radius = f1;
			}
		}

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%d", &d);
		no_slip_particle = (d > 0) ? true : false;

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%d", &particle_initialization_by_file);
		if(particle_initialization_by_file)
		{
			for(i = 0; i < N; i++)
			{
				fgets(buff, bdim, fp);
				sscanf(buff, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
				&f1, &t1, &f, &g, &h, &g1, &h1, &v1, &v2, &v3);

				angular_momentum[i] = f1;
				no_s_gamma[i] = t1;
				radius_CPU[i] = g1;
				mass_CPU[i] = h1;
				p_CPU[i] = make_float3(f, g, h);
				v_CPU[i] = make_float3(v1, v2, v3);
			}
		}
		else
		{
			distribute_particles(MINCD, MAXCD, DIMENSION);
		}
		allocate_GPU_memory();

		fgets(buff, bdim, fp);
		fgets(buff, bdim, fp);
		sscanf(buff, "%s", s);
		strcpy(dir_name, s);

		MIN_CUBE_DIM = MINCD;
		MAX_CUBE_DIM = MAXCD;
	}
}


float get_intersection_point(float time, int p)
{
	return p_CPU[p].x + v_CPU[p].x * time;
}


__device__ int particle_particle_collision(float3 * p, float3 * v, float * radius, int i0, int i1, float * t)
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



__global__ void find_dts(float3 * p, float3 * v, float * radius, float * mass,  Wall * w, // particle data--position, velocity, radius 
			int * tag, int * how_many_p, int * how_many_w, int * what_p_hit, int * what_w_hit, // memory management--what each particle hits and has hit
			int n, int max_complex, int ignore_particle_interaction, float * min_dt) // macros--number of particles, shape of geometry, &c.
{
	float dt, current_min_dt = 0.0;
	int j, k, ok, collides, first_collision, this_particle;
	float3 this_p, this_v;

	first_collision = 1;
	current_min_dt = 20000000;

	this_particle = blockDim.x * blockIdx.x + threadIdx.x;

	if(this_particle < n)
	{
		this_p = p[this_particle];
		this_v = v[this_particle];

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
				if((this_particle == j) || ((tag[max_complex * this_particle] == j) && (tag[max_complex * j] == this_particle)) )ok = 0;
				if(ok > 0)
				{
				  	collides = particle_particle_collision(p, v, radius, this_particle, j, &dt);
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
				collides = w[j].time_to_collision(p[this_particle],v[this_particle],radius[this_particle],&dt);
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

void elastic_particle_particle_collision(int i1, int i2)
{
	float3 n, v1_n, v1_t, v2_n, v2_t;
	float s1_n, s2_n, m1, m2, M;

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

void no_slip_particle_particle_collision(int i1, int i2)
{
	float3 n, t;
	float m,m1,m2,delta,vn1,vt1,vn2,vt2,a1,a2;
	float a11,t11,n11,a22,t22,n22, g1,g2;

	g1 = no_s_gamma[i1]; g2 = no_s_gamma[i2];
	m1 = mass_CPU[i1]; m2 = mass_CPU[i2]; m = m1+m2;

	a1 = angular_momentum[i1]; a2 = angular_momentum[i2];

	delta = 2.0 / ( (1. + 1. / (g1*g1)) / m1 + 
			(1. + 1. / (g2*g2)) / m2);

	n = normalize(p_CPU[i1] - p_CPU[i2]);
	t = make_float3(n.y, n.x, 0.0);

	vn1 = dot(v_CPU[i1], n);
	vt1 = dot(v_CPU[i1], t);
	vn2 = dot(v_CPU[i2], n);
	vt2 = dot(v_CPU[i2], t);

	a11 = 	(1.0 - delta / (m1*g1*g1)) 	* a1 + 
		(delta / (m1*g1)) 		* vt1 + 
		0.0 				* vn1 + 
		(-delta / (m1*g1*g2)) 		* a2 + 
		(-delta / (m1*g1))		* vt2 + 
		0.0 				* vn2;

	t11 = 	(delta / (m1*g1))		* a1 + 
		(1.0 - delta / m1) 		* vt1 + 
		0.0				* vn1 + 
		(delta / (m1*g2))		* a2 + 
		(delta / m1)			* vt2 + 
		0.0 				* vn2;

	n11 = 	0.0 				* a1 + 
		0.0 				* vt1 + 
		(1.0 - 2.0*m2 / m)		* vn1 + 
		0.0 				* a2 + 
		0.0 				* vt2 + 
		(2.0 * m2 / m) 			* vn2;

	a22 = 	(-delta / (m2*g1*g2))	 	* a1 + 
		(delta / (m2*g2)) 		* vt1 + 
		0.0 				* vn1 + 
		(1.0 - delta / (m2*g2*g2)) 	* a2 + 
		(-delta / (m2*g2))		* vt2 + 
		0.0 				* vn2;

	t22 = 	(-delta / (m2*g1))		* a1 + 
		(delta / m2) 			* vt1 + 
		0.0				* vn1 + 
		(-delta / (m2*g2))		* a2 + 
		(1.0 - delta / m2)		* vt2 + 
		0.0 				* vn2;

	n22 = 	0.0 				* a1 + 
		0.0 				* vt1 + 
		(2.0*m1 / m)			* vn1 + 
		0.0 				* a2 + 
		0.0 				* vt2 + 
		(1.0 - 2.0 * m1 / m) 		* vn2;

	angular_momentum[i1] = a11;
	angular_momentum[i2] = a22;

	v_CPU[i1] = t11 * t + n11 * n;
	v_CPU[i2] = t22 * t + n22 * n;

	tag_CPU[max_complex * i1] = i2;
	tag_CPU[max_complex * i2] = i1;
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
	int num_p = 0, num_w = 0;
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
		num_p = how_many_p_CPU[p];
		num_w = how_many_w_CPU[p];
		how_many_p_CPU[p] = how_many_w_CPU[p] = 0;
		for(int i = 0; i < num_p; i++)
			add_recursively_to_complex_event_particles(what_p_CPU[max_complex*p+i], ignored_val);

		complex_event_log[2*complex_colliders  ] = num_p;
		complex_event_log[2*complex_colliders+1] = num_w;
	}
}

bool detect_collision_events()
{
	bool anything_complex_found = false;
	int i, j, k;

	//find global min dt
	dt_step = dt_CPU[0];
	for (i = 1; i < N; i++) if(dt_CPU[i] <= dt_step) dt_step = dt_CPU[i];

	complex_colliders = 0;
	for(i = 0; i < N; i++)
	{
		if(dt_CPU[i] > dt_step) how_many_p_CPU[i] = how_many_w_CPU[i] = 0;

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
		if(     ((p_CPU[i].x*p_CPU[i].x) > (MAX_CUBE_DIM * MAX_CUBE_DIM)) || 
			((p_CPU[i].y*p_CPU[i].y) > (MAX_CUBE_DIM * MAX_CUBE_DIM)) || 
			((p_CPU[i].z*p_CPU[i].z) > (MAX_CUBE_DIM * MAX_CUBE_DIM)) ) 
		{
			printf("Error at time step %d\t physical time %f\t particle %d escaped\n", time_step, total_time, i);
			printf("\tit hit (%d %d) -> %d %d %d\n", how_many_p_CPU[i], how_many_w_CPU[i], 
								tag_CPU[max_complex*i+0], 
								what_w_CPU[max_complex*i+0], 
								what_p_CPU[max_complex*i+1]); 
			printf("\tNew pos: (%f %f %f)\n", p_CPU[i].x, p_CPU[i].y, p_CPU[i].z);
			fflush(stdout);
			exit(1);
		}
	}
}

void print_output_file(FILE *fp, int i, int j, int thisstep, float time, float3 collision_normal)
{
	int k;
	if(thisstep < 0)
	{
		if(DIMENSION < 3)
		{
			for(int jj = 0; jj < N; jj++) fprintf(fp, "%lf %lf %lf ", p_CPU[jj].x, p_CPU[jj].y, radius_CPU[jj]);
				fprintf(fp, "%lf\n", time);
			/*
			float ddt = 0.0;
			int num = int(time/timetol);
			num = (num > 0) ? num : 1;
			for(int jj = 0; jj < num; jj++)
			{
				for(k = 0; k < N; k++) 
				{
					fprintf(fp, "%lf %lf %lf ", 
							p_CPU[k].x + (ddt * v_CPU[k].x), 
							p_CPU[k].y + (ddt * v_CPU[k].y), 
							radius_CPU[k]);
				}
				fprintf(fp, "\n");
				ddt+=timetol;
			}
			*/
		}
	}
	else if(thisstep < 1)
	{
		if(DIMENSION > 2)
		{
			fprintf(fp, "#box dimension\n box, %lf\n", MAX_CUBE_DIM);
			fprintf(fp, "#particle radii\n");
			for(k = 0; k < N; k++) fprintf(fp, "r, %d, %lf, %lf\n", k, radius_CPU[k], mass_CPU[k]);
		}
		else
		{
			FILE *bash_script = NULL;
			bash_script = fopen("edge_file","w");
			for(k = 0; k < num_walls; k++) 
			{
				fprintf(bash_script, "%lf %lf\n%lf %lf\n", 
					walls_CPU[k].endpoints[0].x, 
					walls_CPU[k].endpoints[0].y, 
					walls_CPU[k].endpoints[1].x, 
					walls_CPU[k].endpoints[1].y 
					);
			}
			fclose(bash_script);

			bash_script = fopen("plot_script","w");

			fprintf(bash_script, "end_time=%d\n",j);

			//fprintf(bash_script, "set xrange[%lf:%lf]\nset yrange[%lf:%lf]\nunset key\n", 
			//	-0.5,MAX_CUBE_DIM+0.5,MIN_CUBE_DIM-0.5,MAX_CUBE_DIM+0.5);
			fprintf(bash_script, "set xrange[%lf:%lf]\nset yrange[%lf:%lf]\nunset key\n", 
				MIN_CUBE_DIM-0.5,MAX_CUBE_DIM+0.5,MIN_CUBE_DIM-0.5,MAX_CUBE_DIM+0.5);

			fprintf(bash_script, "set size square\n");
			fprintf(bash_script, "do for [i = 1:end_time]{\n  plot 'edge_file' using 1:2 w l lc 'black', \\\n");
			for(k = 0; k < (N-1); k++) fprintf(bash_script,"    'output.csv' using %d:%d:%d every ::i::i w circles lc %d fillstyle solid, \\\n",3*k+1,3*k+2,3*k+3,k);
			k=(N-1);
			//fprintf(bash_script,"    'output.csv' using %d:%d:%d every ::::i w lines lc %d \n",   3*k+1,3*k+2,3*k+3,k);
			fprintf(bash_script,"    'output.csv' using %d:%d:%d every ::i::i w circles lc %d fillstyle solid \n",   3*k+1,3*k+2,3*k+3,k);

			fprintf(bash_script,"   pause 0.1 \n } ");
			fclose(bash_script);
		}
	}
	else
	{
		if(DIMENSION > 2)
		{
			fprintf(fp, "c, %d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf %lf\n", 
				i, j, time, 
				p_CPU[i].x, p_CPU[i].y, p_CPU[i].z, 
				0.0, 0.0, 0.0,
				v_CPU[i].x, v_CPU[i].y, v_CPU[i].z,
				collision_normal.x, collision_normal.y, collision_normal.z, 
				pressure.latest_value, gas_temperature.latest_value, heat_sum, entropy_sum
			);
		}
	}
}

void n_body()
{

	FILE * out_file;
	FILE * complex_event_log_file;
	char dir[256];

	int i, j, step, w;
	int smart_max_steps = MAX_STEPS;

	float3 v_in, v_out;

	bool complex_collisions_occurred = false;


	//WRITE INITIAL CONDITION TO FILE
	complex_event_log_file = fopen(strcat(strcpy(dir, dir_name), "complex_events_log.txt"), "w");
	out_file = fopen(strcat(strcpy(dir, dir_name), "output.csv"), "w");
	for(i = 0; i < N; i++) print_output_file(out_file, i, smart_max_steps, i, t_tot, collision_normal);

	step = 0;
	while(step++ <= smart_max_steps)
	{
	  	// on GPU - find smallest time step s.t. any particle(s) collide either 
	  	// with each other or a wall and update all particles to that time step
		find_dts<<<grid, block>>>(p_GPU, v_GPU, radius_GPU, mass_GPU, walls_GPU, tag_GPU, how_many_p_GPU, how_many_w_GPU, what_p_GPU, what_w_GPU, N, max_complex, ignore_particle_interaction, dt_GPU);
		errorCheck(step, "find_dts");

		//copy minimum time step and index of corresponding colliding element onto CPU 
		cudaMemcpy( how_many_p_CPU, how_many_p_GPU,             N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy( how_many_w_CPU, how_many_w_GPU,             N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy(     what_p_CPU,     what_p_GPU,max_complex* N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy(     what_w_CPU,     what_w_GPU,max_complex* N * sizeof(int  ), cudaMemcpyDeviceToHost);
		cudaMemcpy(         dt_CPU,         dt_GPU,             N * sizeof(float), cudaMemcpyDeviceToHost);


		// check (and modify tags) if complex collision events occurred
		complex_collisions_occurred = detect_collision_events();

		// if no collisions were detected, we are done. 
		if(dt_step < 0.0)
		{
			printf("\nEarly exit : dt_step = %f < 0 at step %i\n", dt_step, step);
			exit(1);
		}
		t_tot += dt_step;

		print_output_file(out_file,0,0,-4,t_tot,collision_normal);

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
					//end
					if(walls_CPU[w].type == end)
					{
						printf("step %d particle %d escaped through end wall\n", step, i);
						step = smart_max_steps + 2;
						i = N;
						j = num_walls;
						break;
					}
					v_out = walls_CPU[w].resolve_collision(i, v_in);
					tag_CPU[i * max_complex + j] = what_w_CPU[max_complex * i + j];
					v_in = v_out;
				}
				v_CPU[i] = v_out;
				print_output_file(out_file, i, w, step, t_tot, collision_normal);
			}
			else if( how_many_p_CPU[i] > 0)
			{
				j = what_p_CPU[max_complex * i];
				if(i > j) 
				{
					if(no_slip_particle) no_slip_particle_particle_collision(i,j);
					else elastic_particle_particle_collision(i, j);
				}
				print_output_file(out_file, i, j, step, t_tot, collision_normal);
			}
		}

		if( complex_collisions_occurred ) 
		{
			for(i = 0; i < complex_colliders; i++)
			{
				randomize_position(complex_event_particle[i], MIN_CUBE_DIM, MAX_CUBE_DIM);
				fprintf(complex_event_log_file, "%d, ", complex_event_log[i]);
			}
			fprintf(complex_event_log_file, "\n");
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
	for(i = 0; i < N; i++) print_output_file(out_file, i, 0, step, t_tot, collision_normal);

	fclose(out_file);
	fclose(complex_event_log_file);
	printf("%i gas particles, %d steps, %.4f seconds in time\n", N, step, t_tot);
}


int main(int argc, char** argv)
{
	clock_t time_0, time_1;
	FILE *fp;
	char dir[256];

	if(--argc < 1)
	{
		printf("Without input file, reverting to default parameters\n");
		no_input_file = true;
	}
	else
	{
		in_fname = argv[1];
		no_input_file = false;
	}

	time_0 = clock();

	read_input_file_and_set_initial_conditions();

    	n_body();
	time_1 = clock();

	printf("\n Runtime %.5f seconds\n", (float)(time_1 - time_0) / CLOCKS_PER_SEC);
	printf("\n DONE \n");

	fp = fopen(strcat(strcpy(dir, dir_name), "log"), "w");
	fprintf(fp, "N, Nsteps, physical_time, runtime, end_pressure, end_entropy_rate\n%d, %d, %lf, %lf, %lf, %lf", N, MAX_STEPS, t_tot, (float)(time_1 - time_0) / CLOCKS_PER_SEC, pressure.latest_value, entropy_sum / t_tot); 
	fclose(fp);

	return 0;
}
