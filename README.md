# Symplectic Map for Phase-Space Foliation Problem

### Baseline Models and Training
- See ```model.py```
- HenonNet symplectic $T_W$: Sec.1 (train with ```train.py```)
    - Model predictions $x_o, v_o = T_W(x, v)$
    - Target $x_o^* = \sqrt{2J} \sin \omega, v_o^* = \sqrt{2J} \cos \omega$ from Adrian's action-angle pair
    - Optimize $T_W$ to minimize MSE loss $\ell([x_o; v_o], [x_o^*, v_o^*])$
- HenonNetsupQ sympletic $T_W$ and predictor for target quantity $\xi_Q$: Sec.3 (train with ```trainQ.py```)
    - Model predictions $$x_o, v_o = T_W(x, v); \quad J = 0.5(x_o^2 + v_o^2); \quad Q = \xi_Q(J)$$
    - Target $x_o^*, v_o^*, Q^*$
    - Loss $\ell([x_o; v_o], [x_o^*, v_o^*]) + \beta \ell(Q, Q^*)$
