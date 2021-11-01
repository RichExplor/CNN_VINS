import os

class PinholeCamera:

    def __init__(self, fx, fy ,cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0):

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = [k1, k2, p1, p2]

    def distortion(self, p_u):

        k1 = self.d[0]
        k2 = self.d[1]
        p1 = self.d[2]
        p2 = self.d[3]

        mx2_u = p_u[0] * p_u[0]
        my2_u = p_u[1] * p_u[1]
        mxy_u = p_u[0] * p_u[1]
        rho2_u = mx2_u + my2_u
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u

        d_u0 = p_u[0] * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u)
        d_u1 = p_u[1] * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u)

        return (d_u0, d_u1)

    def liftProjective(self, p):
        m_inv_K11 = 1.0 / self.fx
        m_inv_K13 = -self.cx / self.fx
        m_inv_K22 = 1.0 / self.fy
        m_inv_K23 = -self.cy / self.fy

        mx_d = m_inv_K11 * p[0] + m_inv_K13
        my_d = m_inv_K22 * p[1] + m_inv_K23

        n = 8
        d_u = self.distortion((mx_d,my_d))
        mx_u = mx_d - d_u[0]
        my_u = my_d - d_u[1]
        
        for _ in range(n-1):
            d_u = self.distortion((mx_u, my_u))
            mx_u = mx_d - d_u[0]
            my_u = my_d - d_u[1]
        
        return (mx_u, my_u, 1.0)