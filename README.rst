FFrame
======

FFrame is a simple class to discretize a 1D function's
domain and image. Some investigations on performance will also be undertaken.

Dependencies:
    - numpy
    - scipy
    - matplotlib

To do:
    - Write tests and try blocks (*gasp*).
    - Customize plot output.
    - Add direct input of domain.
    - Add a refresh method.
    - Generalize to N-D
    - Add new interpolation schemes.


Questions:
    - What is the operational overhead of implementing an iterative 
    function dicretizer instead of a LUT method?
        - Depends on the division algorithm used in python.
            - One could round to nearest integer before finding nearest allowed value.
                - This would incur floating point errors 
            - One could also spread seeds in the image ensemble and define neighborhoods 
            around seeds where lut values are stored.
            - Apparently python isn't very good at fast division.

    - What is the most energetically efficient implementation?
        - How does it depend on the hardware being used?
        - How does it depend on the language being used?
