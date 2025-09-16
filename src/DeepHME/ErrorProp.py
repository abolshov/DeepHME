import numpy as np

mh = 125.0

class ErrorPropagator():
    """
    computes errors propagated from px, py, pz, E errors to mass errors
    """
    def __init__(self, global_corr_mtrx):
        self._global_corr_mtrx = global_corr_mtrx

    def _energy(self, p3):
        return np.sqrt(mh**2 + np.sum(np.square(p3), axis=1))

    def _mass(self):
        """
        computes mass of X->HH
        """
        assert self.hvv_en is not None and self.hbb_en is not None, "Energies haven't been initialized"
        self.x_en = self.hvv_en + self.hbb_en
        self.x_p3 = self.hvv_p3 + self.hbb_p3
        mx2 = self.x_en**2 - np.sum(np.square(self.x_p3), axis=1)
        negative_mask = mx2 < 0.0
        mx = np.where(negative_mask, -1.0, np.sqrt(np.abs(mx2)))
        return mx

    def _jacobian(self):
        jacobian = None
        if self.hvv_en is None and self.hbb_en is None:
            # case when MX is function of 6 variables (p3 of H->bb and p3 of H->VV)
            self.hvv_en = self._energy(self.hvv_p3)
            self.hbb_en = self._energy(self.hbb_p3)

            self.mx = self._mass()

            # jacobian layout will be:
            # dMx/dpx(H->VV), ..., dMx/dpz(H->VV), dMx/dpx(H->bb), ..., dMx/dpz(H->bb)
            dMx_dPhvv = self.x_p3/self.mx[:, None] - self.x_en[:, None]*self.hvv_p3/(np.concatenate([np.reshape(self.mx*self.hvv_en, (-1, 1))]*3, axis=1))
            dMx_dPhbb = self.x_p3/self.mx[:, None] - self.x_en[:, None]*self.hbb_p3/(np.concatenate([np.reshape(self.mx*self.hbb_en, (-1, 1))]*3, axis=1))
            assert dMx_dPhvv.shape[1] == 3 and dMx_dPhbb.shape[1] == 3
            
            jacobian = np.concatenate([dMx_dPhvv, dMx_dPhbb], axis=1)
            n_ev, n_var = jacobian.shape
            jacobian = jacobian.reshape(n_ev, 1, n_var)
            assert jacobian.shape[-1] == 6
        else:
            # case when MX is function of 8 variables (p3 and E of H->bb and p3 and E of H->VV)
            self.mx = self._mass()

            dMx_dp3 = -1.0*self.x_p3/self.mx[:, None]
            dMx_dE = self.x_en[:, None]/self.mx[:, None]

            jacobian = np.concatenate([dMx_dp3, dMx_dE, dMx_dp3, dMx_dE], axis=1)
            prin(jacobian.shape)
            n_ev, n_var = jacobian.shape
            jacobian = jacobian.reshape(n_ev, 1, n_var)
            assert jacobian.shape[-1] == 8

        return jacobian

    def _event_covar_mtrx(self, errors):
        """
        computes event covariance matrix using errors predicted by net
        renormalizes global correlation matrix by matrix of variances
        """
        max_errors = np.max(errors, axis=0)
        negative_errors = errors < 0.0
        errors = np.where(negative_errors, max_errors[None, :], errors)

        n_events, n_targets = errors.shape
        I = np.eye(n_targets)[None, :, :] # shape (1, n_targets, n_targets)
        std_mtrx = errors[:, :, None]*I # broadcast 

        event_covar_mtrx = np.matmul(np.matmul(std_mtrx, self._global_corr_mtrx), std_mtrx)
        return event_covar_mtrx

    def propagate_errors(self, errors, momentum):
        """
        computes errors on mass from errors on p3 and E
        expected momentum shape is (n_events, 6) or (n_events, 8)
        """
        momentum_shape = momentum.shape
        assert momentum_shape % 2 == 0, f'Momentum shape must be even, got {momentum_shape}'
        hvv_momentum, hbb_momentum = momentum[:, momentum_shape[1]/2:], momentum[:, :momentum_shape[1]/2]

        if hbb_momentum.shape[1] == 4:
            self.hbb_p3 = hbb_momentum[:, :3]
            self.hbb_en = hbb_momentum[:, 3]
        elif hbb_momentum.shape[1] == 3:
            self.hbb_p3 = hbb_momentum[:, :3]
            self.hbb_en = None
        else:
            raise RuntimeError(f'Illegal shape {hbb_momentum.shape[1]} of H->bb momentum')

        if hvv_momentum.shape[1] == 4:
            self.hvv_p3 = hvv_momentum[:, :3]
            self.hvv_en = hvv_momentum[:, 3]
        elif hvv_momentum.shape[1] == 3:
            self.hvv_p3 = hvv_momentum[:, :3]
            self.hvv_en = None
        else:
            raise RuntimeError(f'Illegal shape {hvv_momentum.shape[1]} of H->VV momentum')

        self.jacobian = self._jacobian()
        self.event_covar_mtrx = self._event_covar_mtrx(errors)
        mass_error_sqr = np.matmul(self.jacobian, np.matmul(self.event_covar_mtrx, self.jacobian.transpose(0, 2, 1)))
        assert mass_error_sqr.shape[-1] == 1
        
        negative_mask = mass_error_sqr < 0.0
        mass_error_sqr = np.where(negative_mask, np.max(mass_error_sqr, axis=0), mass_error_sqr)
        return np.sqrt(mass_error_sqr).flatten()
        