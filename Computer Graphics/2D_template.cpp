#include <windows.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <bits/stdc++.h>

using namespace std;

static void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFlush();
}

static void key(unsigned char key, int x, int y)
{
}

static void mouse(int button, int status, int x, int y)
{
}

static void init()
{
    glClearColor(0, 0, 0, 0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1080, 0, 720);
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(1080, 720);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutCreateWindow("GLUT Shapes");
    init();

    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutKeyboardFunc(key);
    glutMainLoop();

    return EXIT_SUCCESS;
}
