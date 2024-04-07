import glfw
from OpenGL.GL import *
import numpy as np

def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # glBegin(GL_LINES)
    # glColor3ub(255,0,0)
    # glVertex2fv(np.array([0.,0.]))
    # glVertex2fv(np.array([1.,0.]))
    # glColor3ub(0,255,0)
    # glVertex2fv(np.array([0.,0.]))
    # glVertex2fv(np.array([0.,1.]))
    # glEnd()
    glBegin(GL_TRIANGLES)
    glColor3ub(255,255,255)
    glVertex2fv(T @ np.array([0.0, 0.5]))
    glVertex2fv(T @ np.array([0.0, 0.0]))
    glVertex2fv(T @ np.array([0.5, 0.0]))
    glEnd()
    
def key_callback(window, key, scancode, action, mods):
    if key==glfw.KEY_A:
        if action==glfw.PRESS:
            print('press a')
        elif action==glfw.RELEASE:
            print('release a')
        elif action==glfw.REPEAT:
            print('repeat a')
        elif key==glfw.KEY_SPACE and action==glfw.PRESS:
            print ('press space: (%d, %d)'%glfw.get_cursor_pos(window))
        

def main():
    if not glfw.init():
        return
    #create windowed mode window and its openGL context
    window = glfw.create_window(640,480,"2D Trans",None,None)
    if not window:
        glfw.terminate()
        return
    
    glfw.set_key_callback(window,key_callback)
    
    #Make the window's context current
    glfw.make_context_current(window)

    #Loop until the user closes the windwo
    while not glfw.window_should_close(window):
        glfw.poll_events()
        T=np.array([[2.,0.],
                    [0.,2.]])
        render(T)
        glfw.swap_buffers(window)
        # t=glfw.get_time()
        # T=np.array([[2.,0.],
        #             [0.,2.]])
        # render(T)
        # nonuniform scale
        # s=np.sin(t)
        # T=np.array([[s,0.],
        #            [0.,s*.5]])
        # rotation
        # th = t
        # T=np.array([[np.cos(th),-np.sin(th)],
        #             [np.sin(th), np.cos(th)]])
        #reflection
        # T=np.array([[-1.,0.],
        #             [0.,1.]])
        # shear
        # a = np.sin(t)
        # T=np.array([[1.,a],
        #             [0.,1.]])
        # identity matrix
        # S=np.array([[1.,0.],
        #             [0.,2.]])
        # th=np.radians(60)
        # R=np.array([[np.cos(th),-np.sin(th)],
        #             [np.sin(th), np.cos(th)]])
        
        # render(R@S)
        
        
    
    glfw.terminate()

if __name__=="__main__":
    main()