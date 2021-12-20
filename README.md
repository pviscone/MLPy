# Machine_learning_project

## TODO
Lista ordinata delle cose da fare
- [x] Trovare metodo più furbo per fare validazione l'attributo di input viene cambiato due volte a ogni step di apprendimento
- [x] Adattare le funzioni a come le vuole Micheli (es. vedi se come errore vuole errore quadratico, errore quadratico medio o altro)
    - [x] MEE
    - [x] MSE
- [x] Implementare funzioni di splitting del dataset esternamente alla classe MLP e Layer
    - [x] hold-out
    - [x] k fold
- [x] Funzione di Rescaling/Standardizzazione dei dati (media 0, varianza 1 oppure range [0,1])
- [x] Implementare funzione per salvare la rete (anche ad es. ogni N epoche (tramite flag booleana))
- [x] Implementare minibatch
- [x] Implementare momento
    - [x] Normale
    - [x] nesterov
- [x] Implementare funzione per creare diverse strategie di learning rate adattivo
- [ ] Implementare criteri di stopping
- [ ] Implementare diverse strategie di inizializzazione dei pesi
- [x] Implementare Grid Search per scelta del modello

### Robe aggiuntive
Da fare solo se si rendono visibilmente necessarie:
- [ ] Cascade correlation
- [ ] Vari tipi di K fold cross validation
- [ ] Algortimi alternativi alla backpropagation: quickprop, RProp,...
- [ ] Dropout
