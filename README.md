# Symplectic Map for Phase-Space Foliation Problem

### Synthetic 1D data
Credit to Adrian Price-Whelan
- Raw data (astro fits file): download [here](https://drive.google.com/file/d/1TZvAeqiuaEhZlkA4qytxu79sqfnrP3E2/view?usp=sharing)
- Train/test data (pickle file storing numpy arrays): download [here](https://drive.google.com/file/d/1V9ml8scWb9De06eSEG5T8qu98agvNQHB/view?usp=sharing) 


### Baseline Models and Training
- See ```model.py```
- HenonNet symplectic $T_W$: Sec.1 (train with ```train_clean.py```)
    - Model predictions $x_o, v_o = T_W(x, v)$
    - Target $x_o^* = \sqrt{2J} \sin \omega, v_o^* = \sqrt{2J} \cos \omega$ from Adrian's action-angle pair
    - Optimize $T_W$ to minimize MSE loss $`\ell([x_o; v_o], [x_o^*, v_o^*])`$
    - Hyper-parameter search: run ```run_grid_search_train_clean.py```
- HenonNetsupQ sympletic $T_W$ and predictor for target quantity $\xi_Q$: Sec.3 (train with ```trainQ.py```)
    - Model predictions $$x_o, v_o = T_W(x, v); \quad J = 0.5(x_o^2 + v_o^2); \quad Q = \xi_Q(J)$$
    - Target $`x_o^*, v_o^*, Q^* `$
    - Loss $` \ell([x_o; v_o], [x_o^*, v_o^*]) + \beta \ell(Q, Q^*) `$
