/** \file
 * Defines interfaces for nodes as used by the KDTree class.
 *
 * \author Martin F. Krafft <krafft@ailab.ch>
 * \date $Date: 2004/11/15 17:40:32 $
 * \version $Revision: 1.6 $
 */

#ifndef INCLUDE_KDTREE_NODE_HPP
#define INCLUDE_KDTREE_NODE_HPP

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS
#include <iostream>
#endif

#include <cstddef>

namespace KDTree
{
  struct _Node_base
  {
    typedef _Node_base* _Base_ptr;

    _Base_ptr _M_parent;
    _Base_ptr _M_left;
    _Base_ptr _M_right;

    _Node_base(_Base_ptr const __PARENT = NULL,
               _Base_ptr const __LEFT = NULL,
               _Base_ptr const __RIGHT = NULL) throw ()
      : _M_parent(__PARENT), _M_left(__LEFT), _M_right(__RIGHT) {}

    static _Base_ptr
    _S_minimum(_Base_ptr __x) throw ()
    {
      while (__x->_M_left) __x = __x->_M_left;
      return __x;
    }
    
    static _Base_ptr
    _S_maximum(_Base_ptr __x) throw ()
    {
      while (__x->_M_right) __x = __x->_M_right;
      return __x;
    }
  };

  template <typename _Val>
    struct _Node : public _Node_base
    {
      using _Node_base::_Base_ptr;
      typedef _Node* _Link_type;

      _Val _M_value;

      _Node(_Val const& __VALUE = _Val(),
            _Base_ptr const __PARENT = NULL,
            _Base_ptr const __LEFT = NULL,
            _Base_ptr const __RIGHT = NULL) throw ()
        : _Node_base(__PARENT, __LEFT, __RIGHT), _M_value(__VALUE) {}

    };

  template <typename _Val, typename _Acc, typename _Cmp>
    class _Node_compare
    {
    public:
      typedef _Node<_Val>* _Link_type;

      _Node_compare(size_t const __DIM) : _M_DIM(__DIM) {} 

      bool
      operator()(_Link_type const& __A, _Link_type const& __B) const
      {
        return _M_cmp(_M_acc(__A->_M_value, _M_DIM),
                      _M_acc(__B->_M_value, _M_DIM));
      }

      bool
      operator()(_Link_type const& __A, _Val const& __B) const
      {
        return _M_cmp(_M_acc(__A->_M_value, _M_DIM), _M_acc(__B, _M_DIM));
      }

      bool
      operator()(_Val const& __A, _Link_type const& __B) const
      {
        return _M_cmp(_M_acc(__A, _M_DIM), _M_acc(__B->_M_value, _M_DIM));
      }

      bool
      operator()(_Val const& __A, _Val const& __B) const
      {
        return _M_cmp(_M_acc(__A, _M_DIM), _M_acc(__B, _M_DIM));
      }

    private:

      size_t const _M_DIM;
      _Acc _M_acc;
      _Cmp _M_cmp;
    };

#ifdef KDTREE_DEFINE_OSTREAM_OPERATORS

  template <typename _Char, typename _Traits, typename _Val>
    std::basic_ostream<_Char, _Traits>&
    operator<<(std::basic_ostream<_Char, _Traits>& __out,
               typename KDTree::_Node<_Val> const& __N) throw ()
    {
      __out << &__N;
      __out << ' ' << __N._M_value;
      __out << "; parent: " << __N._M_parent;
      __out << "; left: " << __N._M_left;
      __out << "; right: " << __N._M_right;
      return __out;
    }

#endif

} // namespace KDTree

#endif // include guard

/* COPYRIGHT --
 *
 * This file is part of libkdtree++, a C++ template KD-Tree sorting container.
 * libkdtree++ is (c) 2004 Martin F. Krafft <krafft@ailab.ch>
 * and distributed under the terms of the Artistic Licence.
 * See the ./COPYING file in the source tree root for more information.
 *
 * THIS PACKAGE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES
 * OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */
