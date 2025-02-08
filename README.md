# Symplectic Map for Phase-Space Foliation Problem

### Baseline Models and Training
- See ```model.py```
- HenonNet symplectic $T_W$: Sec.1 (train with ```train.py```)
    - Model predictions $\hat{x}_o, \hat{v}_o = T_W(x, v)$
    - Target $x_o = \sqrt{2J} \sin \omega, v_o = \sqrt{2J} \cos \omega$ from Adrian's action-angle pair
    - Optimize $T_W$ to minimize MSE loss $\ell([\hat{x}_o; \hat{v}_o], [x_o, v_o])$
- HenonNetsupQ sympletic $T_W$ and predictor for target quantity $\xi_Q$: Sec.3 (train with ```trainQ.py```)
    - Model predictions $$\hat{x}_o, \hat{v}_o = T_W(x, v); \quad \widehat{J} = 0.5(\hat{x}_o^2 + \hat{v}_o^2); \quad \widehat{Q} = \xi_Q(\widehat{J})$$
    - Target $x_o, v_o, Q$
    - Loss $\ell([\hat{x}_o; \hat{v}_o], [x_o, v_o]) + \beta \ell(\widehat{Q}, Q)$
