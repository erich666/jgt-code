/** \file
 * Defines the interface for the Accessor class.
 *
 * \author Martin F. Krafft <krafft@ailab.ch>
 * \date $Date: 2004/11/15 17:40:32 $
 * \version $Revision: 1.5 $
 */

#ifndef INCLUDE_KDTREE_ACCESSOR_HPP
#define INCLUDE_KDTREE_ACCESSOR_HPP

#include <cstddef>

namespace KDTree
{
  template <typename _Val>
    struct Accessor
    {
      typedef _Val value_type;
      typedef typename value_type::value_type subvalue_type;

      virtual subvalue_type
      operator()(value_type const&, size_t const) const = 0;
      virtual ~Accessor() {}
    };

  template <typename _Val>
    struct _Bracket_accessor : public Accessor<_Val>
    {
      typedef _Val value_type;
      typedef typename value_type::value_type subvalue_type;
      
      subvalue_type
      operator()(value_type const& V, size_t const N) const throw ()
      {
        return V[N];
      }
    };

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
