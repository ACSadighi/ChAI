.. default-domain:: chpl

.. module:: OrderedDict

OrderedDict
===========
**Usage**

.. code-block:: chapel

   use OrderedDict;


or

.. code-block:: chapel

   import OrderedDict;

.. record:: dict : serializable

   .. attribute:: type keyType

   .. attribute:: type valType

   .. attribute:: var table: map(keyType, valType)

   .. attribute:: var order: list(keyType)

   .. method:: proc init(in table: map(?keyType, ?valType), in order: list(keyType))

   .. method:: proc init(in table: map(?keyType, ?valType))

   .. method:: proc init(type keyType, type valType)

   .. method:: proc size: int

   .. itermethod:: iter keys(): keyType

   .. itermethod:: iter values(): valType

   .. itermethod:: iter ref these()

   .. method:: proc ref insert(key: keyType, in value: valType)

