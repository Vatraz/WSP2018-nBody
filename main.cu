#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <windows.h>
#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>    
#include <timer.h>      
#include <chrono>
#include <helper_cuda.h>         


#define REFRESH_DELAY		10 //ms
#define ZMIEKCZANIE			0.001f
#define G					0.66742f
#define KROK				0.005f

using namespace std;
using namespace std::chrono;


// Wymiary okna
const int okno_szerokosc = 800;
const int okno_wysokosc = 500;

int liczba_cial;
const int liczba_watkow = 256;
int liczba_blokow;
float czas_symulacji = 0.0;
float mnoznik_predkosci;

// OpenGL Vertex Buffer Object
GLuint vbo;

// Inicjalizacja obiektow/ buforow obiektow
float4 *d_obiekty_v;
float4 *h_obiekty_v;
float4 *h_obiekty_wsp_m;

// licznik FPS 
high_resolution_clock::time_point timer1;
high_resolution_clock::time_point timer2;
int frameCount = 0;
float avg_FPS = 0.0f;
int limit_FPS = 100;  

// prototypy funkcji
bool initGL(int *argc, char **argv);
void stworz_tablice();
void stworz_VBO(GLuint *vbo);
void uruchomienie_kernela();
void losowanie(float4 *obiekty_v, float4 *obiekty_wsp_m, int n);
void cpu_test();
void cleanup();

// callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0;
float rotate_y = 0.0;
float translate_z = -20.0;

// wyliczenie sily wypadkowej dzialajacej na cialo
__device__ float3 sily(float4 *obiekty_wsp_m, float4 *obiekty_v, float4 ten_obiekt, int liczba_cial) {
	
	float3 r;
	float3 a = { 0.0f, 0.0f, 0.0f };
	float kwadrat_wektorow;
	float mianownik;

	for (int i = 0; i < liczba_cial; i++) {
		// aktualna pozycja
		r.x = obiekty_wsp_m[i].x - ten_obiekt.x;
		r.y = obiekty_wsp_m[i].y - ten_obiekt.y;
		r.z = obiekty_wsp_m[i].z - ten_obiekt.z;

		// liczenie odleglosci
		kwadrat_wektorow = r.x*r.x + r.y*r.y + r.z*r.z + ZMIEKCZANIE;
		mianownik = 1.f / kwadrat_wektorow;
		
		// obliczanie przyspieszen - kieunku wektora
		float s = G * ten_obiekt.w * mianownik;
		a.x += r.x * s;
		a.y += r.y * s;
		a.z += r.z * s;
	}
	return a;
}

// wyliczenie pozycji ciala
__global__ void pozycje(float4 *obiekty_wsp_m, float4 *obiekty_v, int liczba_cial, float mnoznik_predkosci)
{	
	__shared__  float4 obiekty_wsp_SHARED[liczba_watkow];
	int numer_obiektu = blockIdx.x * blockDim.x + threadIdx.x;
	if (numer_obiektu < liczba_cial) {
		obiekty_wsp_SHARED[threadIdx.x] = obiekty_wsp_m[numer_obiektu];

		__syncthreads();

		// liczenie przyspieszen
		float3 a;
		a = sily(obiekty_wsp_m, obiekty_v, obiekty_wsp_SHARED[threadIdx.x], liczba_cial);
		obiekty_v[numer_obiektu].x += a.x * KROK * mnoznik_predkosci;
		obiekty_v[numer_obiektu].y += a.y * KROK * mnoznik_predkosci;
		obiekty_v[numer_obiektu].z += a.z * KROK * mnoznik_predkosci;
									
		__syncthreads();

		// nowe pozycje obiektow
		obiekty_wsp_SHARED[threadIdx.x].x += obiekty_v[numer_obiektu].x * KROK;
		obiekty_wsp_SHARED[threadIdx.x].y += obiekty_v[numer_obiektu].y * KROK;
		obiekty_wsp_SHARED[threadIdx.x].z += obiekty_v[numer_obiektu].z * KROK;

		__syncthreads();

		// wpisanie nowej pozycji
		obiekty_wsp_m[numer_obiektu] = obiekty_wsp_SHARED[threadIdx.x];
	}
}

int main(int argc, char **argv)
{
	printf("Liczba cial: ");
	scanf("%i", &liczba_cial);
	printf("Mnoznik predkosci: ");
	scanf("%f", &mnoznik_predkosci);
	liczba_blokow = ((liczba_cial-1) / liczba_watkow) + 1;

	// openGL init
	initGL(&argc, argv);

	// stworzenie VBO- empty vertex buffer object, inicjalizacja 
	stworz_tablice();
	stworz_VBO(&vbo);
	
	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutCloseFunc(cleanup);

	// cuda
	high_resolution_clock::time_point gpu1 = high_resolution_clock::now();
	uruchomienie_kernela();
	cudaDeviceSynchronize();
	high_resolution_clock::time_point gpu2 = high_resolution_clock::now();
	duration<double> gput = duration_cast<duration<double>>(gpu2 - gpu1);

	// cpu
	high_resolution_clock::time_point cpu1 = high_resolution_clock::now();
	cpu_test();
	high_resolution_clock::time_point cpu2 = high_resolution_clock::now();
	duration<double> cput = duration_cast<duration<double>>(cpu2 - cpu1);

	// porownanie czasow
	printf("\nCzas potrzebny na wyliczenie jednego kroku symulacji: ");
	printf("\nCPU: %E", cput.count());
	printf("\nGPU: %E\n", gput.count());
	printf("\nPrzyspieszenie x%f\n\n", cput.count()/ gput.count());

	// glowna petla
	glutMainLoop();
	return 0;
}

// wykonanie obliczen na gpu
void uruchomienie_kernela()
{
	// mapuje bufor OpenGL do CUDA
	float4 *dptr;
	cudaGLMapBufferObject((void**)&dptr, vbo);

	pozycje << <liczba_blokow, liczba_watkow >> >(dptr, d_obiekty_v, liczba_cial, mnoznik_predkosci);

	// unmap vpo
	cudaGLUnmapBufferObject(vbo);
}

// Petla animacji
void display()
{	
	// uruchomienie licznika FPS
	if (frameCount == 0 )
		timer1 = high_resolution_clock::now();

	// wykonanie obliczen na GPU
	uruchomienie_kernela();
	
	// ustawienie macierzy widoku
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// renderowanie 
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(0.5, 0.5, 1.0);
	glDrawArrays(GL_POINTS, 0, liczba_cial);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();

	// obsluga licznika FPS
	czas_symulacji += KROK;
	frameCount++;
	if (frameCount >= limit_FPS)
	{
		timer2 = high_resolution_clock::now();
		duration<double> czasFps = duration_cast<duration<double>>(timer2 - timer1);
		avg_FPS = frameCount / czasFps.count() ;
		frameCount = 0;
		char fps[256];
		sprintf(fps, "nBody: %2.2f fps", avg_FPS);
		glutSetWindowTitle(fps);
	}
	
}

// inicjalizacja openGL
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(okno_szerokosc, okno_wysokosc);
	glutCreateWindow("nBody:");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, okno_szerokosc, okno_wysokosc);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)okno_szerokosc / (GLfloat)okno_wysokosc, 0.1, 100.0);

	SDK_CHECK_ERROR_GL();

	return true;
}

// tablice zawierajace polozenie i predksoc cial
void stworz_tablice() 
{
	int rozmiar_obiektow = 4 * liczba_cial * sizeof(float);

	// alokancja elegancka
	h_obiekty_v = (float4*)malloc(rozmiar_obiektow);
	h_obiekty_wsp_m = (float4*)malloc(rozmiar_obiektow);
	cudaMalloc(&d_obiekty_v, rozmiar_obiektow);

	// zerowanie predkosci, losowanie polozenia i masy
	losowanie(h_obiekty_v, h_obiekty_wsp_m, liczba_cial);

	// kopiowanie wylosowanych danych do device
	cudaMemcpy(d_obiekty_v, h_obiekty_v, rozmiar_obiektow, cudaMemcpyHostToDevice);
}

// tworzenie OpenGL Vertex Buffer Object
void stworz_VBO(GLuint *vbo)
{
	int rozmiar_obiektow = 4 * liczba_cial * sizeof(float);
	
	assert(vbo);

	// tworzenie bufora
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	glBufferData(GL_ARRAY_BUFFER, rozmiar_obiektow, h_obiekty_wsp_m, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGLRegisterBufferObject(*vbo));

	SDK_CHECK_ERROR_GL();
}

// w ktorej czesci znajduja sie ciala
int znak() {
	int plusminus = rand() & 5;

	if (plusminus > 2)
		plusminus = 1;
	else
		plusminus = -1;
	return plusminus;
}

// losowanie polozen i predkosci cial
void losowanie(float4 *obiekty_v, float4 *obiekty_wsp_m, int n)
{
	float x; //e10
	float y; //e10
	float z; //e10

	for (int i = 0; i < n; i++) {
		x = znak() * 5.f * (rand() / (float)RAND_MAX); //e10
		y = znak() * 5.f * (rand() / (float)RAND_MAX); //e10
		z = znak() * 5.f * (rand() / (float)RAND_MAX); //e10
		obiekty_v[i] = make_float4(0.f, 0.f, 0.f, 1.f);
		obiekty_wsp_m[i] = make_float4(x, y, z, 2.f);
	}
}

// sily liczone na CPU
float3 CPUsily(float4 *obiekty_wsp_m, int indeks)
{
	float3 r;
	float3 a = { 0.0f, 0.0f, 0.0f };
	float kwadrat_wektorow;
	float distSqr3;
	float mianownik;

	for (int i = 0; i < liczba_cial; i++) {
		// aktualna pozycja
		r.x = obiekty_wsp_m[i].x - obiekty_wsp_m[indeks].x;
		r.y = obiekty_wsp_m[i].y - obiekty_wsp_m[indeks].y;
		r.z = obiekty_wsp_m[i].z - obiekty_wsp_m[indeks].z;

		// liczenie odleglosci
		kwadrat_wektorow = r.x*r.x + r.y*r.y + r.z*r.z + ZMIEKCZANIE;
		distSqr3 = kwadrat_wektorow;
		mianownik = 1.f / distSqr3;

		// obliczanie przyspieszen - kieunku wektora

		float s = G * obiekty_wsp_m[i].w * mianownik;
		a.x += r.x * s;
		a.y += r.y * s;
		a.z += r.z * s;
	}
	return a;
}

// pozycje cial liczone na CPU
void CPUpozycje(float4 *obiekty_wsp_m, float4 *obiekty_v)
{
	// liczenie przyspieszen
	float3 a;
	for (int i = 0; i < liczba_cial; i++) {
		a = CPUsily(obiekty_wsp_m, i);
		obiekty_v[i].x += a.x * KROK;
		obiekty_v[i].y += a.y * KROK;
		obiekty_v[i].z += a.z * KROK;
	}

	// nowe pozycje obiektow
	for (int i = 0; i < liczba_cial; i++) {
		obiekty_wsp_m[i].x += obiekty_v[i].x * KROK;
		obiekty_wsp_m[i].y += obiekty_v[i].y * KROK;
		obiekty_wsp_m[i].z += obiekty_v[i].z * KROK;
	}
}

// test wydajnosci CPU
void cpu_test() {
	CPUpozycje(h_obiekty_wsp_m, h_obiekty_v);
}

// usuniecie vbo
void deleteVBO(GLuint *vbo)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

// czyszczenie po zamknieciu
void cleanup()
{
	deleteVBO(&vbo);
	cudaFree(d_obiekty_v);
	free(h_obiekty_v);
	free(h_obiekty_wsp_m);
}

// obsluga klawiatury
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27): //esc
		glutDestroyWindow(glutGetWindow());
		return; 
	case (87): //W
		mnoznik_predkosci = mnoznik_predkosci / 2;
		return;
	case (119): //w
		mnoznik_predkosci = mnoznik_predkosci / 2;
		return;
	case (115): //s
		mnoznik_predkosci = mnoznik_predkosci * 2;
		return;
	case (83): //S
		mnoznik_predkosci = mnoznik_predkosci * 2;
		return;
	}
}

// obsluga myszy
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

// sterowanie
void motion(int x, int y)
{
	double dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.05f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

// glut timer
void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}