import glfw
from OpenGL.GL import *
import numpy as np

global glPrimitive

def render():
    global glPrimitive
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    
    glBegin(glPrimitive)
    glColor3ub(255,255,255)
    tharr=np.linspace(0,2*np.pi,13)
    for th in tharr[:-1]:
        glVertex2fv(np.array([np.cos(th),np.sin(th)]))
    glEnd()

def key_callback(window,key,scancode,action,mods):
    global glPrimitive
    if key==glfw.KEY_1:
        glPrimitive=GL_POINTS
    elif key==glfw.KEY_2:
        glPrimitive=GL_LINES
    elif key==glfw.KEY_3:
        glPrimitive=GL_LINE_STRIP
    elif key==glfw.KEY_4:
        glPrimitive=GL_LINE_LOOP
    elif key==glfw.KEY_5:
        glPrimitive=GL_TRIANGLES
    elif key==glfw.KEY_6:
        glPrimitive=GL_TRIANGLE_STRIP
    elif key==glfw.KEY_7:
        glPrimitive=GL_TRIANGLE_FAN
    elif key==glfw.KEY_8:
        glPrimitive=GL_QUADS
    elif key==glfw.KEY_9:
        glPrimitive=GL_QUAD_STRIP
    elif key==glfw.KEY_0:
        glPrimitive=GL_POLYGON
        
def main():
    global glPrimitive
    if not glfw.init():
        return
    window = glfw.create_window(480,480,"2020090791-2-1",None,None)
    if not window:
        glfw.terminate()
        return
    glfw.set_key_callback(window,key_callback)
    glfw.make_context_current(window)
    glPrimitive=GL_LINE_LOOP
    while not glfw.window_should_close(window):
        glfw.poll_events()
        render()
        glfw.swap_buffers(window)
        
    glfw.terminate()
    
if __name__=="__main__":
    main()