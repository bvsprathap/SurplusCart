"""
reports/report_generator.py

Report module — deterministic Python only. No LLM calls in this file.
"""

from __future__ import annotations

import os
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import uuid

import folium
import folium.plugins
import pandas as pd
from PIL import Image
import polyline

from data.data_model import CareHome, SimulationDay, Store, Volunteer, WorldConfig, DailyFoodItem, FoodCatalogItem
from tools.logger import get_message_log
from tools.models import (
    Delivery,
    DispatchStats,
    NegotiationResult,
    NegotiationTurn,
    Order,
    OrderLineItem,
)
from tools.constraint_tools import hard_constraint_filter, single_store_candidate, StockLedger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPORTS_DIR = Path(__file__).parent / "output"
_NON_VEGETARIAN_ITEMS: frozenset[str] = frozenset({"chicken", "eggs"})


def _ensure_output_dir() -> Path:
    """Create reports/output/ if it doesn't exist. Returns the directory Path."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return _REPORTS_DIR


# ---------------------------------------------------------------------------
# Background Image Splitter
# ---------------------------------------------------------------------------

def split_background_image():
    """Split Back_ground_2.png into header.png and footer.png on first run."""
    parent_dir = Path(__file__).parent
    
    src = None
    # Check possible directory names and spaces/underscores
    for folder in ["Assets", "assets"]:
        for name in ["Back ground 2.png", "Back_ground_2.png"]:
            p = parent_dir / folder / name
            if p.exists():
                src = p
                break
        if src:
            break
            
    if not src:
        src = parent_dir / "assets" / "Back_ground_2.png"
        
    assets_dir = parent_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    header_path = assets_dir / "header.png"
    footer_path = assets_dir / "footer.png"
    
    if not header_path.exists() or not footer_path.exists():
        if src.exists():
            try:
                img = Image.open(src)
                img.crop((0, 0, 1600, 130)).save(header_path)
                img.crop((0, 620, 1600, 900)).save(footer_path)
            except Exception as e:
                print(f"[Warning] Failed to split background image: {e}")


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _store_map(world: WorldConfig) -> Dict[str, Store]:
    return {s.store_id: s for s in world.stores}


def _care_home_map(world: WorldConfig) -> Dict[str, CareHome]:
    return {ch.care_home_id: ch for ch in world.care_homes}


def _volunteer_map(world: WorldConfig) -> Dict[str, Volunteer]:
    return {v.volunteer_id: v for v in world.volunteers}


def _order_map(orders: List[Order]) -> Dict[str, Order]:
    return {o.order_id: o for o in orders}


def _items_for_delivery(delivery: Delivery, order_map: Dict[str, Order]) -> List[OrderLineItem]:
    """Collect all accepted line items across all orders in this delivery."""
    lines: List[OrderLineItem] = []
    for oid in delivery.order_ids:
        order = order_map.get(oid)
        if order:
            lines.extend(order.items)
    return lines


def _urgent_in_delivery(delivery: Delivery, order_map: Dict[str, Order]) -> List[str]:
    """Collect all urgent_essential_items across all orders in this delivery."""
    urgent: List[str] = []
    for oid in delivery.order_ids:
        order = order_map.get(oid)
        if order:
            urgent.extend(order.urgent_essential_items)
    return list(dict.fromkeys(urgent))  # deduplicate, preserve order


# Distance helper for skip reason estimation
def approx_distance_minutes(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Standard speed approximation: 30 km/h in Chennai city traffic (0.5 km per minute).
    # 1 degree lat is ~111 km. 1 degree lon is ~108 km.
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 108.0
    dist_km = math.sqrt(dlat*dlat + dlon*dlon)
    return dist_km * 2.0  # 2 mins per km


def estimate_vol_time(vol: Volunteer, store: Store, care_home_coords: List[List[float]]) -> float:
    time_vol_to_store = approx_distance_minutes(vol.latitude, vol.longitude, store.latitude, store.longitude)
    current_loc = [store.latitude, store.longitude]
    time_store_to_ch = 0.0
    for ch_coord in care_home_coords:
        time_store_to_ch += approx_distance_minutes(current_loc[0], current_loc[1], ch_coord[0], ch_coord[1])
        current_loc = ch_coord
    return time_vol_to_store + time_store_to_ch


def get_items_summary(items: List[OrderLineItem]) -> str:
    grouped_qty = {}
    grouped_unit = {}
    for li in items:
        if li.accepted_quantity > 0:
            grouped_qty[li.item] = grouped_qty.get(li.item, 0.0) + li.accepted_quantity
            grouped_unit[li.item] = li.unit
            
    sorted_items = sorted(grouped_qty.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_items[:3]
    summary_parts = [f"{name}: {qty:.1f} {grouped_unit[name]}" for name, qty in top_3]
    if len(sorted_items) > 3:
        summary_parts.append(f"+ {len(sorted_items) - 3} more")
    return ", ".join(summary_parts) if summary_parts else "—"


def get_fallback_reason(d: Delivery, payload_kg: float, has_urgent: bool, world: WorldConfig, sim_day: SimulationDay) -> str:
    if d.method == "volunteer":
        return "—"
    
    available_vols = []
    vol_states = {v.volunteer_id: v.available for v in sim_day.volunteers}
    for v in world.volunteers:
        if vol_states.get(v.volunteer_id, False):
            available_vols.append(v)
            
    if d.method == "store_truck":
        if has_urgent:
            return "Urgent items on delivery forced fallback to store truck"
        if not available_vols:
            return "All volunteers unavailable today"
        has_capacity = any(v.capacity_kg >= payload_kg for v in available_vols)
        if not has_capacity:
            return "All available volunteers have insufficient capacity"
        return "All volunteers exceed the 120-minute route time budget"
        
    if d.method == "commercial":
        # Direct commercial check: if this delivery contains needs_commercial items
        # Check if the store truck is available
        store_state = next((s for s in sim_day.stores if s.store_id == d.store_id), None)
        truck_avail = store_state.truck_status.available if (store_state and store_state.truck_status) else False
        
        if not d.order_ids:
            return "Direct commercial assignment (4th store required)"
        if not truck_avail:
            return "Store truck unavailable — routed to commercial"
        return "Delivery payload exceeds store truck capacity"
    return "—"


# ---------------------------------------------------------------------------
# SECTION 1 — Delivery table
# ---------------------------------------------------------------------------

def generate_delivery_table(
    deliveries: List[Delivery],
    orders: List[Order],
    world: WorldConfig,
) -> tuple[str, str]:
    """
    Produce a console-ready table of all deliveries.
    """
    sm = _store_map(world)
    chm = _care_home_map(world)
    vm = _volunteer_map(world)
    om = _order_map(orders)

    rows = []
    for d in deliveries:
        store = sm.get(d.store_id)
        store_name = store.name if store else d.store_id

        if d.method == "volunteer" and d.volunteer_id:
            vol = vm.get(d.volunteer_id)
            method_str = f"Volunteer: {vol.name if vol else d.volunteer_id}"
        elif d.method == "store_truck":
            method_str = "Store Truck"
        else:
            method_str = "Commercial"

        for oid in d.order_ids:
            order = om.get(oid)
            if not order:
                continue

            ch = chm.get(order.care_home_id)
            care_home_str = ch.name if ch else order.care_home_id

            items_str = ", ".join(
                f"{li.item}: {li.accepted_quantity:.1f} {li.unit}"
                for li in order.items
                if li.accepted_quantity > 0
            ) or "—"

            urgent_names = [
                li.item for li in order.items 
                if li.accepted_quantity > 0 and li.item.lower() in {u.lower() for u in order.urgent_essential_items}
            ]
            urgent_str = "★ " + ", ".join(urgent_names) if urgent_names else "—"

            rows.append({
                "Delivery ID": d.delivery_id[:8] + "…",
                "Store": store_name,
                "Care Home(s)": care_home_str,
                "Items": items_str,
                "Method": method_str,
                "Pickup": d.pickup_time or "—",
                "Urgent": urgent_str,
            })

    if not rows:
        output = "(No deliveries to display)"
        print(output)
        return output, ""

    df = pd.DataFrame(rows)
    for col in ("Items", "Care Home(s)"):
        df[col] = df[col].apply(lambda x: textwrap.fill(str(x), width=40))

    header = "\n" + "=" * 100 + "\n  DELIVERY TABLE\n" + "=" * 100 + "\n"
    table_str = df.to_string(index=False, max_colwidth=50)
    output = header + table_str + "\n" + "=" * 100 + "\n"

    try:
        print(output)
    except UnicodeEncodeError:
        pass
    return output, df.to_html(index=False, border=0, classes='table table-striped', justify='center')


# ---------------------------------------------------------------------------
# SECTION 2 — Map
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "volunteer": "green",
    "store_truck": "blue",
    "commercial": "orange",
}
_CHENNAI_CENTER = [13.0827, 80.2707]

def generate_map(
    deliveries: List[Delivery],
    orders: List[Order],
    world: WorldConfig,
    run_id: str,
) -> tuple[str, str]:
    """
    Generate a Folium HTML map of all deliveries and save to
    reports/output/map_{run_id}.html.
    """
    _ensure_output_dir()
    filepath = str(_REPORTS_DIR / f"map_{run_id}.html")

    sm = _store_map(world)
    chm = _care_home_map(world)
    vm = _volunteer_map(world)
    om = _order_map(orders)

    m = folium.Map(location=[13.03, 80.245], zoom_start=12, tiles="CartoDB positron")

    for store in world.stores:
        folium.Marker(
            location=[store.latitude, store.longitude],
            popup=folium.Popup(store.name, parse_html=True),
            tooltip=store.name,
            icon=folium.Icon(color="blue", icon="shopping-cart", prefix="fa"),
        ).add_to(m)

    for ch in world.care_homes:
        folium.Marker(
            location=[ch.latitude, ch.longitude],
            popup=folium.Popup(ch.name, parse_html=True),
            tooltip=ch.name,
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(m)

    assigned_vols = {d.volunteer_id for d in deliveries if d.method == "volunteer" and d.volunteer_id}

    for vol in world.volunteers:
        is_assigned = vol.volunteer_id in assigned_vols
        
        folium.CircleMarker(
            location=[vol.latitude, vol.longitude],
            radius=5 if not is_assigned else 7,
            color="#A0522D" if is_assigned else "grey",
            fill=True,
            fill_color="#A0522D" if is_assigned else "grey",
            fill_opacity=0.9 if is_assigned else 0.7,
            tooltip=f"{vol.name} (Assigned)" if is_assigned else vol.name,
            popup=folium.Popup(f"{vol.name} ({vol.vehicle_type})", parse_html=True),
        ).add_to(m)

    method_counters = {"volunteer": 0, "store_truck": 0, "commercial": 0}

    route_list_items = []

    for delivery in deliveries:
        color = _METHOD_COLORS.get(delivery.method, "grey")
        method_counters[delivery.method] += 1
        seq_num = method_counters[delivery.method]
        
        store = sm.get(delivery.store_id)
        if not store:
            continue
        store_coord = [store.latitude, store.longitude]
        
        ch_coords = []
        for oid in delivery.order_ids:
            order = om.get(oid)
            if order:
                ch = chm.get(order.care_home_id)
                if ch:
                    ch_coords.append([ch.latitude, ch.longitude])

        vol_coord = None
        if delivery.method == "volunteer" and delivery.volunteer_id:
            vol = vm.get(delivery.volunteer_id)
            vol_coord = [vol.latitude, vol.longitude] if vol else None
            waypoints = (
                ([vol_coord] if vol_coord else []) +
                [store_coord] +
                ch_coords
            )
        else:
            waypoints = [store_coord] + ch_coords

        if len(waypoints) < 2:
            continue

        name_str = delivery.method.replace('_', ' ').title()
        if delivery.method == "volunteer" and vol_coord:
            name_str = vol.name.split()[0]
        route_list_items.append(f"{seq_num}. {name_str}")

        has_polyline = False
        path_coords = waypoints
        if getattr(delivery, 'polyline', None):
            try:
                decoded = polyline.decode(delivery.polyline)
                if decoded:
                    path_coords = decoded
                    has_polyline = True
            except Exception:
                pass

        # Split into legs for arrows
        legs = []
        if delivery.method == "volunteer" and vol_coord:
            def dist_sq(p1, p2):
                return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            min_dist = float('inf')
            store_idx = 0
            for i, pt in enumerate(path_coords):
                d = dist_sq(pt, store_coord)
                if d < min_dist:
                    min_dist = d
                    store_idx = i
            if 0 < store_idx < len(path_coords) - 1:
                legs.append(path_coords[:store_idx+1])
                legs.append(path_coords[store_idx:])
            else:
                legs.append(path_coords)
        else:
            legs.append(path_coords)

        dash = "10 5" if delivery.method == "commercial" else None
        tooltip_text = f"#{seq_num} | {delivery.method.replace('_', ' ').title()} | Store: {store.name}"

        poly_line = folium.PolyLine(
            locations=path_coords,
            color=color,
            weight=3,
            opacity=0.8,
            dash_array=dash,
            tooltip=tooltip_text,
        )
        poly_line.add_to(m)
        
        for leg in legs:
            leg_poly = folium.PolyLine(locations=leg, opacity=0, weight=0)
            leg_poly.add_to(m)
            folium.plugins.PolyLineTextPath(
                leg_poly,
                "\u25BA",  # ►
                center=True,
                offset=7,
                attributes={"fill": color, "font-size": "15px", "font-weight": "bold"}
            ).add_to(m)

        # Sequential milestones list for the route
        milestones = []
        if delivery.method == "volunteer" and vol_coord:
            milestones.append(vol_coord)
        milestones.append(store_coord)
        milestones.extend(ch_coords)

        # Progressive sequential milestone matching
        milestone_indices = []
        last_idx = 0
        for ms in milestones:
            min_d = float('inf')
            best_idx = last_idx
            for idx in range(last_idx, len(path_coords)):
                pt = path_coords[idx]
                d = (pt[0] - ms[0])**2 + (pt[1] - ms[1])**2
                if d < min_d:
                    min_d = d
                    best_idx = idx
            milestone_indices.append(best_idx)
            last_idx = best_idx

        # Isolate delivery legs (excluding volunteer-to-store pickup leg)
        start_ms_idx = 1 if (delivery.method == "volunteer" and vol_coord) else 0
        delivery_legs = []
        for m_i in range(start_ms_idx, len(milestone_indices) - 1):
            s_idx = milestone_indices[m_i]
            e_idx = milestone_indices[m_i + 1]
            if s_idx < e_idx:
                delivery_legs.append(path_coords[s_idx:e_idx + 1])
            else:
                # Handle fallback if indices overlap or collide
                delivery_legs.append([path_coords[s_idx], path_coords[e_idx]])

        # Place sequence midpoint markers on each active delivery leg
        for leg in delivery_legs:
            if len(leg) >= 2:
                total_dist = 0.0
                segment_dists = []
                for i in range(len(leg) - 1):
                    p1 = leg[i]
                    p2 = leg[i + 1]
                    d = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    segment_dists.append(d)
                    total_dist += d
                
                if total_dist > 0:
                    target_dist = total_dist / 2.0
                    curr_dist = 0.0
                    mid_lat = None
                    mid_lon = None
                    for i in range(len(leg) - 1):
                        d = segment_dists[i]
                        if curr_dist + d >= target_dist:
                            rem = target_dist - curr_dist
                            f = rem / d if d > 0 else 0.0
                            mid_lat = leg[i][0] + f * (leg[i + 1][0] - leg[i][0]) + 0.003
                            mid_lon = leg[i][1] + f * (leg[i + 1][1] - leg[i][1])
                            break
                        curr_dist += d
                    
                    if mid_lat is None or mid_lon is None:
                        mid_lat = leg[-1][0] + 0.003
                        mid_lon = leg[-1][1]
                else:
                    mid_lat = leg[0][0] + 0.003
                    mid_lon = leg[0][1]
            else:
                mid_lat = leg[0][0] + 0.003
                mid_lon = leg[0][1]

            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=f"""<div style="background-color: {color}; color: white; 
                                border-radius: 50%; width: 20px; height: 20px; 
                                display: flex; justify-content: center; align-items: center; 
                                font-size: 12px; font-weight: bold; border: 1px solid white;">
                            {seq_num}</div>""",
                    icon_anchor=(10, 10),
                ),
                tooltip=tooltip_text
            ).add_to(m)

    route_list_html = "<b>Routes</b><br>" + "<br>".join(route_list_items)

    legend_html = f"""
    <div style="position:fixed; bottom:30px; right:30px; z-index:1000;
                background:white; padding:15px; border:1px solid #aaa;
                border-radius:6px; font-size:13px; line-height:1.8;
                display:flex; flex-direction:row; gap:20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
      <div>
          <b>Delivery Methods</b><br>
          <span style="color:green;">&#9644;</span> Volunteer Route<br>
          <span style="color:blue;">&#9644;</span> Store Truck Route<br>
          <span style="color:orange;">&#9644;</span> Commercial Route<br>
      </div>
      <div>
          <b>Locations</b><br>
          <span style="color:#A0522D;">&#9679;</span> Assigned Volunteer<br>
          <span style="color:grey;">&#9679;</span> Idle Volunteer<br>
          <i class="fa fa-shopping-cart" style="color:blue"></i> Store<br>
          <i class="fa fa-home" style="color:green"></i> Care Home<br>
      </div>
      <div>
          {route_list_html}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    use_api_links = os.environ.get("RUNNING_ON_CLOUD_RUN") or os.environ.get("SERVED_VIA_API")
    summary_link = "/" if use_api_links else "latest_summary.html"
    report_link = "/report" if use_api_links else "latest_report.html"
    buttons_html = f"""
    <div style="position:fixed; top:20px; right:20px; z-index:1000; display:flex; gap:10px;">
      <a href="{summary_link}"
         style="background-color:#000517; color:#04D8D9; border:1px solid #04D8D9;
                padding:10px 20px; font-family:'Segoe UI', Arial, sans-serif; font-size:14px;
                font-weight:bold; text-decoration:none; border-radius:4px;
                box-shadow: 0 0 8px rgba(4, 216, 217, 0.4); transition: all 0.3s ease;"
         onmouseover="this.style.backgroundColor='#04D8D9'; this.style.color='#000517'; this.style.boxShadow='0 0 12px #01F3F4';"
         onmouseout="this.style.backgroundColor='#000517'; this.style.color='#04D8D9'; this.style.boxShadow='0 0 8px rgba(4, 216, 217, 0.4)';">
        View Summary
      </a>
      <a href="{report_link}"
         style="background-color:#000517; color:#04D8D9; border:1px solid #04D8D9;
                padding:10px 20px; font-family:'Segoe UI', Arial, sans-serif; font-size:14px;
                font-weight:bold; text-decoration:none; border-radius:4px;
                box-shadow: 0 0 8px rgba(4, 216, 217, 0.4); transition: all 0.3s ease;"
         onmouseover="this.style.backgroundColor='#04D8D9'; this.style.color='#000517'; this.style.boxShadow='0 0 12px #01F3F4';"
         onmouseout="this.style.backgroundColor='#000517'; this.style.color='#04D8D9'; this.style.boxShadow='0 0 8px rgba(4, 216, 217, 0.4)';">
        View Report
      </a>
    </div>
    """
    m.get_root().html.add_child(folium.Element(buttons_html))

    map_html = m.get_root().render()
    
    if not os.environ.get("RUNNING_ON_CLOUD_RUN"):
        m.save(filepath)
        latest_map_path = str(_REPORTS_DIR / "map.html")
        m.save(latest_map_path)
        print(f"\n[Map] Saved to: {filepath} and {latest_map_path}\n")
    
    return filepath, map_html


# ---------------------------------------------------------------------------
# SECTION 3 — Negotiation report
# ---------------------------------------------------------------------------

_DIVIDER = "\n" + "─" * 80 + "\n"

def generate_negotiation_report(
    negotiation_results: List[NegotiationResult],
    orders: List[Order],
    world: WorldConfig,
) -> str:
    """
    Full negotiation transcript and outcome per care home.
    """
    chm = _care_home_map(world)
    orders_by_ch: Dict[str, List[Order]] = {}
    for o in orders:
        orders_by_ch.setdefault(o.care_home_id, []).append(o)

    result_map = {r.care_home_id: r for r in negotiation_results}
    all_home_ids = [ch.care_home_id for ch in world.care_homes]

    parts: List[str] = []
    header = "\n" + "=" * 80 + "\n  NEGOTIATION REPORT\n" + "=" * 80

    for ch_id in all_home_ids:
        ch = chm.get(ch_id)
        ch_name = ch.name if ch else ch_id
        negotiates = ch.negotiates_via_a2a if ch else True

        result = result_map.get(ch_id)
        section_lines = [
            f"\nCARE HOME: {ch_name}",
            f"Negotiates via A2A: {'Yes' if negotiates else 'No (auto-accept)'}",
        ]

        if result is None:
            section_lines.append("  [No negotiation result recorded for this home]")
            parts.append("\n".join(section_lines))
            continue

        if result.offer_message:
            section_lines.append("\nOffer Sent:")
            section_lines.append("  " + result.offer_message.replace("\n", "\n  "))


        if result.status == "rejected":
            section_lines.append(f"STATUS: ✗ REJECTED")
            section_lines.append(f"Rejection message: {result.rejection_message or '—'}")
            if result.negotiation_transcript:
                section_lines.append("\nTranscript:")
                for turn in result.negotiation_transcript:
                    _format_turn(turn, section_lines)
            parts.append("\n".join(section_lines))
            continue

        if not negotiates:
            section_lines.append("STATUS: ✓ AUTO-ACCEPTED REMAINDER")
            section_lines.append("\nItems auto-accepted:")
            for item in result.agreed_items:
                section_lines.append(
                    f"  • {item.item}: {item.accepted_quantity:.1f} {item.unit}"
                )
        else:
            section_lines.append("STATUS: ✓ AGREED")

            if result.negotiation_transcript:
                section_lines.append("\nTranscript:")
                for turn in result.negotiation_transcript:
                    _format_turn(turn, section_lines)

            section_lines.append("\nItems offered vs agreed:")
            section_lines.append(f"  {'Item':<25} {'Offered':>10} {'Agreed':>10} {'Unit':<8}")
            section_lines.append(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8}")
            for item in result.agreed_items:
                section_lines.append(
                    f"  {item.item:<25} {item.offered_quantity:>10.0f} "
                    f"{item.accepted_quantity:>10.0f} {item.unit:<8}"
                )

        if result.urgent_item_names:
            section_lines.append(
                f"\nUrgent items flagged: ★ {', '.join(result.urgent_item_names)}"
            )

        ch_orders = orders_by_ch.get(ch_id, [])
        all_arriving = []
        all_deferred = []
        all_messages = []
        for order in ch_orders:
            if order.final_notice:
                fn = order.final_notice
                if fn.get("arriving_today"):
                    all_arriving.extend(fn["arriving_today"])
                if fn.get("deferred"):
                    all_deferred.extend(fn["deferred"])
                if fn.get("message") and fn["message"] not in all_messages:
                    all_messages.append(fn["message"])
        
        if all_messages:
            section_lines.append(f"\nToday's Delivery details:")
            for msg in all_messages:
                section_lines.append(f"  {msg}")

        parts.append("\n".join(section_lines))

    body = _DIVIDER.join(parts)
    console_output = header + "\n" + body + "\n" + "=" * 80 + "\n"
    try:
        print(console_output)
    except UnicodeEncodeError:
        pass
    

    # Return HTML
    html_parts = []
    for ch_id in all_home_ids:
        ch = chm.get(ch_id)
        ch_name = ch.name if ch else ch_id
        negotiates = ch.negotiates_via_a2a if ch else True
        result = result_map.get(ch_id)
        
        status_class = "status-agreed"
        status_text = "Agreed"
        if result is None:
            status_text = "No Result"
            status_class = "status-none"
        elif result.status == "rejected":
            status_text = "REJECTED"
            status_class = "status-rejected"
        elif not negotiates:
            status_text = "AUTO-ACCEPTED"
            status_class = "status-auto"
            
        html_part = f"""
        <div class="negotiation-card" style="margin-top: 15px; padding: 15px; background: #000F2E; border: 1px solid rgba(4, 216, 217, 0.2); border-radius: 6px;">
            <h4 style="margin: 0 0 10px 0; color: #04D8D9; font-size: 16px;">{ch_name} 
              <span class="{status_class}" style="font-size: 12px; margin-left: 10px; padding: 2px 8px; border-radius: 4px; background: rgba(4, 216, 217, 0.1); color: #04D8D9;">{status_text}</span>
            </h4>
            <p style="font-size: 13px; margin: 5px 0;">Negotiates via A2A: {"Yes" if negotiates else "No (auto-accept)"}</p>
        """
        if result:
            if result.offer_message:
                html_part += f"<div style='margin-top: 10px; font-size: 13px; color: #EAFBFF; padding: 10px; border-left: 3px solid #04D8D9; background: #000517;'><b>Offer Sent:</b><br/>{result.offer_message.replace(chr(10), '<br/>')}</div>"

            if result.status == "rejected":
                html_part += f"<p style='color: #01F3F4; font-weight: bold; margin: 10px 0;'>Status: ✗ REJECTED</p>"
                html_part += f"<p style='color: #087C81; font-style: italic; margin: 5px 0;'>Rejection message: {result.rejection_message or '—'}</p>"
            
            if result.negotiation_transcript:
                html_part += """
                <div style="background: #000517; padding: 10px; border-radius: 4px; font-family: Courier, monospace; font-size: 12px; margin-top: 10px; margin-bottom: 10px; border-left: 3px solid #087C81;">
                    <b style="color: #04D8D9;">Negotiation Transcript:</b><br/>
                """
                for turn in result.negotiation_transcript:
                    speaker_label = "SYSTEM" if turn.speaker == "system" else "CARE HOME"
                    action_label = turn.action.replace("_", " ").upper()
                    if turn.item:
                        if turn.quantity is not None:
                            detail_str = f" [{turn.item}: {turn.quantity:.1f}]"
                        else:
                            detail_str = f" [{turn.item}]"
                    else:
                        detail_str = ""
                    html_part += f"<span style='color: #EAFBFF;'>Turn {turn.turn_number:>2} | {speaker_label} | {action_label}{detail_str}</span><br/>"
                html_part += "</div>"
                
            if result.status != "rejected" and result.agreed_items:
                html_part += "<table style='width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 5px; border: 1px solid rgba(4, 216, 217, 0.2);'>"
                html_part += "<tr style='background: #087C81; color: #EAFBFF; text-align: left;'><th>Item</th><th>Offered</th><th>Agreed</th><th>Unit</th></tr>"
                for item in result.agreed_items:
                    html_part += f"<tr style='border-bottom: 1px solid rgba(4, 216, 217, 0.1); background: #000F2E;'><td>{item.item}</td><td>{item.offered_quantity:.1f}</td><td>{item.accepted_quantity:.1f}</td><td>{item.unit}</td></tr>"
                html_part += "</table>"
                
            if result.urgent_item_names:
                html_part += f"<p style='margin-top: 10px; font-size: 13px; color: #01F3F4;'><b>Urgent items flagged:</b> ★ {', '.join(result.urgent_item_names)}</p>"
                
            ch_orders = orders_by_ch.get(ch_id, [])
            all_arriving = []
            all_deferred = []
            all_messages = []
            for order in ch_orders:
                if order.final_notice:
                    fn = order.final_notice
                    if fn.get("arriving_today"):
                        all_arriving.extend(fn["arriving_today"])
                    if fn.get("deferred"):
                        all_deferred.extend(fn["deferred"])
                    if fn.get("message") and fn["message"] not in all_messages:
                        all_messages.append(fn["message"])
            
            if all_messages:
                html_part += f"<div style='margin-top: 10px; font-size: 12px; color: #04D8D9; border-top: 1px dashed rgba(4, 216, 217, 0.2); padding-top: 5px;'>"
                formatted_messages = '<br/>'.join(msg.replace('\\n', '<br/>').replace('\n', '<br/>') for msg in all_messages)
                html_part += f"<b>Today's Delivery details:</b><br/>{formatted_messages}<br/>"
                html_part += "</div>"
        else:
            html_part += "<p style='color: #087C81; font-style: italic;'>[No negotiation result recorded for this home]</p>"
        html_part += "</div>"
        html_parts.append(html_part)
        
    return "\n".join(html_parts)


def _format_turn(turn: NegotiationTurn, lines: List[str]) -> None:
    """Append a single negotiation turn as a formatted line."""
    speaker = "  SYSTEM   " if turn.speaker == "system" else "  CARE HOME"
    action = turn.action.replace("_", " ").upper()
    detail = ""
    if turn.item:
        detail += f" [{turn.item}"
        if turn.quantity is not None:
            detail += f": {turn.quantity:.1f}"
        detail += "]"
    lines.append(f"  Turn {turn.turn_number:>2} | {speaker} | {action}{detail}")


# ---------------------------------------------------------------------------
# SECTION 4 — Audit report
# ---------------------------------------------------------------------------

def generate_audit_report(
    sim_day: SimulationDay,
    world: WorldConfig,
    dispatch_stats: DispatchStats,
) -> str:
    """
    Per-store inventory audit (full vs pushed, held-back items).
    """
    sm = _store_map(world)
    catalog_map = {c.name.lower(): c for c in world.catalog}

    lines = [
        "\n" + "=" * 80,
        "  AUDIT REPORT",
        "=" * 80,
        "",
        "━━━ 4a. STORE INVENTORY AUDIT ━━━",
    ]

    for store_state in sim_day.stores:
        store = sm.get(store_state.store_id)
        store_name = store.name if store else store_state.store_id

        full_count = len(store_state.full_inventory)
        pushed_names = {item.name.lower() for item in store_state.pushed_inventory}
        pushed_count = len(store_state.pushed_inventory)

        held_back = [
            item for item in store_state.full_inventory
            if item.name.lower() not in pushed_names
        ]

        lines.append(f"\n  {store_name}")
        lines.append(f"  Full inventory : {full_count} items")
        lines.append(f"  Pushed today   : {pushed_count} items")

        if store_state.pushed_inventory:
            lines.append("  ┌─ PUSHED (near-expiry / threshold crossed):")
            for item in sorted(store_state.pushed_inventory, key=lambda x: x.days_to_expiry):
                lines.append(
                    f"  │  {item.name:<28} {item.quantity:>7.1f} {item.unit:<6} "
                    f"(expires in {item.days_to_expiry}d)"
                )

        if held_back:
            lines.append("  └─ HELD BACK (below push threshold):")
            for item in sorted(held_back, key=lambda x: x.days_to_expiry, reverse=True):
                cat = catalog_map.get(item.name.lower())
                threshold = cat.push_threshold_days if cat else "?"
                lines.append(
                    f"     {item.name:<28} expires in {item.days_to_expiry}d "
                    f"(threshold: {threshold}d)"
                )

    lines += [
        "",
        "━━━ 4b. DISPATCH EFFICIENCY SUMMARY ━━━",
        "",
        f"  Total deliveries       : {dispatch_stats.total_deliveries}",
        f"  ├─ Volunteer           : {dispatch_stats.volunteer_assigned}",
        f"  ├─ Store truck         : {dispatch_stats.store_truck_assigned}",
        f"  └─ Commercial          : {dispatch_stats.commercial_assigned}",
        "",
        f"  Volunteers unavailable : {dispatch_stats.volunteers_unavailable}",
        f"  Detours bundled        : {dispatch_stats.detours_bundled}",
        f"  Urgent forced fallback : {dispatch_stats.urgent_items_forced_fallback}",
    ]

    if dispatch_stats.commercial_assigned > 0:
        lines.append(
            f"\n  ⚠  {dispatch_stats.commercial_assigned} commercial pickup(s) — "
            "check route viability with operations team."
        )

    lines.append("\n" + "=" * 80 + "\n")

    output = "\n".join(lines)
    try:
        print(output)
    except UnicodeEncodeError:
        pass
    

    # Return HTML
    html_parts = []
    html_parts.append("""
    <div class="audit-report-container" style="margin-top: 25px; padding: 20px; background: #000F2E; border: 1px solid rgba(4, 216, 217, 0.4); border-radius: 8px;">
        <h3 style="color: #04D8D9; font-size: 16px; margin-top: 0; margin-bottom: 20px; font-family: 'Century Gothic Bold', sans-serif;">Store Inventory & Dispatch Audit</h3>
    """)
    
    html_parts.append('<div style="margin-bottom: 20px;">')
    html_parts.append('<h4 style="color: #04D8D9; font-size: 14px; margin-bottom: 10px;">Store Inventory Audit</h4>')
    for store_state in sim_day.stores:
        store = sm.get(store_state.store_id)
        store_name = store.name if store else store_state.store_id
        
        pushed_names = {item.name.lower() for item in store_state.pushed_inventory}
        held_back = [
            item for item in store_state.full_inventory
            if item.name.lower() not in pushed_names
        ]
        
        html_parts.append(f"""
        <div style="margin-top: 15px; padding: 10px; background: #000517; border-radius: 4px; border-left: 3px solid #087C81;">
            <b style="color: #EAFBFF; font-size: 13px;">{store_name}</b><br/>
            <span style="font-size: 12px; color: #EAFBFF; opacity: 0.8;">Full inventory: {len(store_state.full_inventory)} items | Pushed today: {len(store_state.pushed_inventory)} items</span><br/>
        """)
        
        if store_state.pushed_inventory:
            html_parts.append('<div style="margin-top: 5px; font-size: 12px; color: #EAFBFF;"><b>Pushed items:</b>')
            for item in sorted(store_state.pushed_inventory, key=lambda x: x.days_to_expiry):
                html_parts.append(f'<br/>&nbsp;&nbsp;&bull; {item.name}: {item.quantity:.1f} {item.unit} (expires in {item.days_to_expiry}d)')
            html_parts.append('</div>')
            
        if held_back:
            html_parts.append('<div style="margin-top: 5px; font-size: 12px; color: #EAFBFF; opacity: 0.7;"><b>HELD BACK:</b>')
            for item in sorted(held_back, key=lambda x: x.days_to_expiry, reverse=True):
                cat = catalog_map.get(item.name.lower())
                threshold = cat.push_threshold_days if cat else "?"
                html_parts.append(f'<br/>&nbsp;&nbsp;&bull; {item.name} (expires in {item.days_to_expiry}d, threshold: {threshold}d)')
            html_parts.append('</div>')
            
        html_parts.append('</div>')
    html_parts.append('</div>')
    
    html_parts.append(f"""
    <div class="audit-summary-box" style="margin-top: 15px; padding: 15px; background: #000517; border: 1px solid rgba(4, 216, 217, 0.2); border-radius: 6px;">
        <h4 style="margin: 0 0 10px 0; color: #04D8D9; font-size: 14px;">Dispatch Efficiency Summary</h4>
        <ul style="list-style: none; padding-left: 0; font-size: 13px; line-height: 1.6; margin: 0;">
            <li><b>Total Deliveries:</b> {dispatch_stats.total_deliveries}</li>
            <li>&nbsp;&nbsp;&bull; Volunteer Assigned: {dispatch_stats.volunteer_assigned}</li>
            <li>&nbsp;&nbsp;&bull; Store Truck: {dispatch_stats.store_truck_assigned}</li>
            <li>&nbsp;&nbsp;&bull; Commercial (Dunzo): {dispatch_stats.commercial_assigned}</li>
            <li><b>Volunteers Unavailable:</b> {dispatch_stats.volunteers_unavailable}</li>
            <li><b>Detours Bundled:</b> {dispatch_stats.detours_bundled}</li>
            <li><b>Urgent Forced Fallback:</b> {dispatch_stats.urgent_items_forced_fallback}</li>
        </ul>
    """)
    if dispatch_stats.commercial_assigned > 0:
        html_parts.append(
            f"<p style='color: #01F3F4; font-size: 12px; margin-top: 10px;'>⚠ {dispatch_stats.commercial_assigned} commercial pickup(s) recorded.</p>"
        )
    html_parts.append("</div></div>")
    
    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# MAIN HTML POST-PROCESSORS
# ---------------------------------------------------------------------------

def highlight_rejected_rows(table_html: str, ch_rows: List[dict]) -> str:
    lines = table_html.split("<tr>")
    new_lines = [lines[0]]
    for line in lines[1:]:
        is_rejected = False
        for r in ch_rows:
            if r["Status"] == "Rejected" and f"<td>{r['Care Home']}</td>" in line:
                is_rejected = True
                break
        if is_rejected:
            new_lines.append(f'<tr class="rejected-row" style="background-color: #087C81 !important; color: #EAFBFF;">{line}')
        else:
            new_lines.append(f'<tr>{line}')
    return "".join(new_lines)


def highlight_volunteer_rows(table_html: str, vol_rows: List[dict]) -> str:
    lines = table_html.split("<tr>")
    new_lines = [lines[0]]
    for line in lines[1:]:
        vol_name = None
        for r in vol_rows:
            if f"<td>{r['Volunteer']}</td>" in line:
                vol_name = r['Volunteer']
                status_raw = r['_status_raw']
                break
        if vol_name:
            if "Assigned" in status_raw:
                new_lines.append(f'<tr class="vol-assigned" style="border-left: 4px solid #087C81;">{line}')
            elif "Unavailable" in status_raw:
                new_lines.append(f'<tr class="vol-unavailable" style="opacity: 0.6;">{line}')
            else:
                new_lines.append(f'<tr>{line}')
        else:
            new_lines.append(f'<tr>{line}')
    return "".join(new_lines)


def highlight_delivery_rows(table_html: str, del_rows: List[dict]) -> str:
    lines = table_html.split("<tr>")
    new_lines = [lines[0]]
    for line in lines[1:]:
        del_id = None
        for r in del_rows:
            if f"<td>{r['Delivery ID']}</td>" in line:
                del_id = r['Delivery ID']
                method_raw = r['_method_raw']
                break
        if del_id:
            if method_raw == "volunteer":
                new_lines.append(f'<tr class="route-volunteer" style="border-left: 4px solid #04D8D9;">{line}')
            elif method_raw == "store_truck":
                new_lines.append(f'<tr class="route-truck" style="border-left: 4px solid #087C81;">{line}')
            elif method_raw == "commercial":
                new_lines.append(f'<tr class="route-commercial" style="border-left: 4px solid #01F3F4;">{line}')
            else:
                new_lines.append(f'<tr>{line}')
        else:
            new_lines.append(f'<tr>{line}')
    return "".join(new_lines)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def generate_summary_page(
    deliveries: List[Delivery],
    orders: List[Order],
    negotiation_results: List[NegotiationResult],
    dispatch_stats: DispatchStats,
    world: WorldConfig,
    sim_day: SimulationDay,
    run_id: str
) -> Tuple[str, str]:
    # Math
    catalog_map = {c.name.lower(): c for c in world.catalog}
    food_rescued_kg = 0.0
    for o in orders:
        for li in o.items:
            qty = li.accepted_quantity
            cat = catalog_map.get(li.item.lower())
            w = cat.approx_weight_kg if cat else 1.0
            food_rescued_kg += qty * w
            
    meals_served = round(food_rescued_kg / 0.5)
    ghg_avoided = round(food_rescued_kg * 2.5)
    
    served_home_ids = {o.care_home_id for d in deliveries for o_id in d.order_ids for o in orders if o.order_id == o_id}
    homes_served_count = len(served_home_ids)
    residents_served = 0
    chm = _care_home_map(world)
    for ch_id in served_home_ids:
        ch = chm.get(ch_id)
        if ch:
            residents_served += ch.resident_count

    # Operational KPIs
    total_pushed_kg = 0.0
    for store_state in sim_day.stores:
        for item in store_state.pushed_inventory:
            qty = item.quantity
            cat = catalog_map.get(item.name.lower())
            w = cat.approx_weight_kg if cat else 1.0
            total_pushed_kg += qty * w
            
    waste_prevention_rate = (food_rescued_kg / total_pushed_kg * 100) if total_pushed_kg > 0 else 0.0
    fulfilled_orders = sum(len(d.order_ids) for d in deliveries)
    delivery_success_rate = (fulfilled_orders / len(orders) * 100) if orders else 0.0
    
    available_vols = sum(1 for v in sim_day.volunteers if v.available)
    volunteer_part_str = f"{dispatch_stats.volunteer_assigned} of {available_vols}"
    
    urgent_flagged = sum(len(o.urgent_essential_items) for o in orders)
    urgent_delivered = 0
    for d in deliveries:
        for oid in d.order_ids:
            for o in orders:
                if o.order_id == oid:
                    urgent_delivered += len(o.urgent_essential_items)
                    
    urgent_fulfilled_str = f"{urgent_delivered} of {urgent_flagged}"
    urgent_color = "#01F3F4" if urgent_delivered == urgent_flagged and urgent_flagged > 0 else "#EAFBFF"

    # Per-Home Impact Table
    home_rows = []
    for ch in world.care_homes:
        if ch.care_home_id in served_home_ids:
            h_orders = [o for d in deliveries for oid in d.order_ids for o in orders if o.order_id == oid and o.care_home_id == ch.care_home_id]
            h_items = 0
            h_kg = 0.0
            h_urgent = 0
            for ho in h_orders:
                for li in ho.items:
                    h_items += li.accepted_quantity
                    qty = li.accepted_quantity
                    cat = catalog_map.get(li.item.lower())
                    w = cat.approx_weight_kg if cat else 1.0
                    h_kg += qty * w
                h_urgent += len(ho.urgent_essential_items)
                
            h_methods = list({d.method.replace('_', ' ').title() for d in deliveries for oid in d.order_ids for ho in h_orders if ho.order_id == oid})
            home_rows.append({
                "Care Home": ch.name,
                "Residents": ch.resident_count,
                "Items Received": f"{h_items:.1f}",
                "kg Received": f"{h_kg:.1f}",
                "Meals (est.)": f"{round(h_kg/0.5)}",
                "Urgent Items Honored": h_urgent,
                "Delivery Method(s)": ", ".join(h_methods)
            })
        else:
            home_rows.append({
                "Care Home": ch.name,
                "Residents": ch.resident_count,
                "Items Received": "—",
                "kg Received": "—",
                "Meals (est.)": "—",
                "Urgent Items Honored": "—",
                "Delivery Method(s)": "Declined today"
            })
    
    df_home = pd.DataFrame(home_rows)
    table_home_html = df_home.to_html(classes="data-table", border=0, index=False, escape=False)
    
    # Narrative
    narrative_parts = []
    store_count = len({d.store_id for d in deliveries})
    narrative_parts.append(f"Today, {food_rescued_kg:.1f} kg of surplus food from {store_count} Chennai stores reached {homes_served_count} care homes, providing an estimated {meals_served} meals to {residents_served} residents.")
    if urgent_flagged > 0:
        narrative_parts.append(f"All {urgent_delivered} urgent essential requests were fulfilled.")
    
    truck_count = dispatch_stats.store_truck_assigned
    comm_count = dispatch_stats.commercial_assigned
    vol_count = dispatch_stats.volunteer_assigned
    narrative_parts.append(f"Deliveries were completed via {vol_count} volunteer trips, {truck_count} store-truck runs, and {comm_count} commercial pickups.")
    narrative_parts.append(f"An estimated {ghg_avoided} kg of CO2e emissions were avoided by preventing this food from going to waste.")
    narrative_html = " ".join(narrative_parts)

    use_api_links = os.environ.get("RUNNING_ON_CLOUD_RUN") or os.environ.get("SERVED_VIA_API")
    report_link = "/report" if use_api_links else "latest_report.html"
    map_link = "/map" if use_api_links else f"map_{run_id}.html"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SurplusCart - Daily Impact Summary</title>
  <style>
    body {{
      background-color: #000517;
      color: #EAFBFF;
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }}
    
    header, footer {{
      width: 100%;
      max-width: 1600px;
      margin: 0 auto;
    }}
    header img, footer img {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .title-block {{
      text-align: center;
      margin: 20px auto 40px;
    }}
    .title-block h1 {{
      font-family: 'Century Gothic Bold', 'Arial Bold', sans-serif;
      color: #04D8D9;
      font-size: 28px;
      margin: 0 0 10px 0;
      text-shadow: 0 0 8px #01F3F4, 0 0 16px #04D8D9;
    }}
    .title-block p {{
      font-size: 14px;
      color: #EAFBFF;
      margin: 0;
    }}

    .content-container {{
      width: 90%;
      max-width: 1200px;
      margin: 0 auto;
      flex-grow: 1;
    }}

    .card {{
      background-color: #000F2E;
      border: 1px solid rgba(4, 216, 217, 0.4);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 30px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }}
    
    .card h2 {{
      font-family: 'Century Gothic Bold', 'Arial Bold', sans-serif;
      color: #04D8D9;
      font-size: 18px;
      margin-top: 0;
      margin-bottom: 20px;
      text-shadow: 0 0 8px #01F3F4, 0 0 16px #04D8D9;
    }}

    .kpi-row {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 30px;
    }}
    .kpi-card {{
      flex: 1;
      background-color: #000F2E;
      border: 1px solid rgba(4, 216, 217, 0.4);
      border-radius: 8px;
      padding: 20px;
      text-align: center;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }}
    .kpi-value {{
      font-family: 'Century Gothic Bold', 'Arial Bold', sans-serif;
      color: #01F3F4;
      font-size: 48px;
      margin: 10px 0;
      text-shadow: 0 0 8px rgba(1, 243, 244, 0.5);
    }}
    .kpi-label {{
      font-size: 14px;
      color: #EAFBFF;
      font-weight: bold;
    }}
    .kpi-sub {{
      font-size: 12px;
      color: #087C81;
      margin-top: 5px;
    }}

    .kpi-row-small .kpi-card {{
      padding: 15px;
    }}
    .kpi-row-small .kpi-value {{
      font-size: 28px;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 20px;
      font-size: 14px;
      color: #EAFBFF;
    }}
    .data-table th {{
      background-color: #087C81;
      color: #EAFBFF;
      text-align: left;
      padding: 10px 12px;
      font-weight: 600;
    }}
    .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(4, 216, 217, 0.4);
    }}
    .data-table tr:nth-child(even) {{
      background-color: #000F2E;
    }}
    .data-table tr:nth-child(odd) {{
      background-color: #001433;
    }}

    .btn-container {{
      display: flex;
      justify-content: center;
      gap: 20px;
      margin: 40px 0;
    }}
    .btn-action {{
      background-color: #04D8D9;
      color: #000517;
      padding: 12px 32px;
      font-family: 'Century Gothic Bold', sans-serif;
      font-size: 16px;
      font-weight: bold;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
      transition: all 0.3s ease;
      box-shadow: 0 0 8px rgba(4, 216, 217, 0.4);
    }}
    .btn-action:hover {{
      background-color: #01F3F4;
      box-shadow: 0 0 12px #01F3F4;
    }}
  </style>
</head>
<body>
  <header>
    <img src="../assets/header.png" alt="Header Image" onerror="this.style.display='none';"/>
  </header>

  <div class="title-block">
    <h1>SurplusCart &mdash; Daily Impact Summary</h1>
    <p>Run ID: {run_id} | Simulated Date: Today</p>
  </div>

  <div class="content-container">
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-value">{food_rescued_kg:.1f}</div>
        <div class="kpi-label">Food Rescued (kg)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{meals_served}</div>
        <div class="kpi-label">Estimated Meals Served</div>
        <div class="kpi-sub">Est. at 0.5 kg per meal (FAO convention)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{homes_served_count}</div>
        <div class="kpi-label">Care Homes Served</div>
        <div class="kpi-sub">reaching {residents_served} residents</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{ghg_avoided}</div>
        <div class="kpi-label">Estimated CO2e Avoided (kg)</div>
        <div class="kpi-sub">Est. at 2.5 kg CO2e per kg food waste avoided (WRAP-derived factor)</div>
      </div>
    </div>

    <div class="kpi-row kpi-row-small">
      <div class="kpi-card">
        <div class="kpi-value">{waste_prevention_rate:.1f}%</div>
        <div class="kpi-label">Waste Prevention Rate</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{delivery_success_rate:.1f}%</div>
        <div class="kpi-label">Delivery Success Rate</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{volunteer_part_str}</div>
        <div class="kpi-label">Volunteer Participation</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value" style="color: {urgent_color};">{urgent_fulfilled_str}</div>
        <div class="kpi-label">Urgent Items Fulfilled</div>
      </div>
    </div>

    <div class="card">
      <h2>Narrative Summary</h2>
      <p style="line-height: 1.6;">{narrative_html}</p>
    </div>

    <div class="card">
      <h2>Per-Home Impact</h2>
      {table_home_html}
    </div>

    <div class="btn-container">
      <a href="{report_link}" target="_blank" class="btn-action">View Detailed Report</a>
      <a href="{map_link}" target="_blank" class="btn-action">View Delivery Map</a>
    </div>
  </div>

  <footer>
    <img src="../assets/footer.png" alt="Footer Image" onerror="this.style.display='none';"/>
  </footer>
</body>
</html>"""
    summary_filepath = str(_REPORTS_DIR / f"summary_{run_id}.html")
    latest_filepath = str(_REPORTS_DIR / "latest_summary.html")

    if not os.environ.get("RUNNING_ON_CLOUD_RUN"):
        with open(summary_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        with open(latest_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
            
    return summary_filepath, html_content

async def generate_full_report(
    deliveries: List[Delivery],
    orders: List[Order],
    negotiation_results: List[NegotiationResult],
    dispatch_stats: DispatchStats,
    world: WorldConfig,
    sim_day: SimulationDay,
    run_id: str,
) -> dict:
    """
    Full end-to-end HTML report generation.
    """
    _ensure_output_dir()
    split_background_image()

    sm = _store_map(world)
    chm = _care_home_map(world)
    vm = _volunteer_map(world)
    om_lookup = _order_map(orders)

    # 1. Store Inventory Tables
    store_tables_html = []
    catalog_map = {c.name.lower(): c for c in world.catalog}
    alloc_map = {}
    for o in orders:
        ch = chm.get(o.care_home_id)
        ch_name = ch.name if ch else o.care_home_id
        for li in o.items:
            if li.accepted_quantity > 0:
                key = (o.store_id, li.item.lower())
                alloc_map.setdefault(key, []).append(f"{ch_name} ({li.accepted_quantity:.1f})")

    for store_state in sim_day.stores:
        store = sm.get(store_state.store_id)
        store_name = store.name if store else store_state.store_id
        pushed_names = {item.name.lower() for item in store_state.pushed_inventory}
        pushed_qty_map = {item.name.lower(): item.quantity for item in store_state.pushed_inventory}

        pushed_items = [item for item in store_state.full_inventory if item.name.lower() in pushed_names]
        pushed_items.sort(key=lambda x: x.days_to_expiry)

        non_pushed_items = [item for item in store_state.full_inventory if item.name.lower() not in pushed_names]
        non_pushed_items.sort(key=lambda x: x.days_to_expiry)

        sorted_items = pushed_items + non_pushed_items

        rows_inv = []
        for item in sorted_items:
            item_lower = item.name.lower()
            cat = catalog_map.get(item_lower)
            category = "Perishable" if (cat and cat.is_perishable) else "Non-Perishable"

            is_pushed = item_lower in pushed_names
            pushed_str = "Yes" if is_pushed else "No"
            pushed_qty = pushed_qty_map.get(item_lower, 0.0) if is_pushed else 0.0
            pushed_qty_str = f"{pushed_qty:.1f}" if is_pushed else ""

            allocated_list = alloc_map.get((store_state.store_id, item_lower), [])
            allocated_str = ", ".join(allocated_list) if allocated_list else "—"

            allocated_qty = sum(li.accepted_quantity for o in orders if o.store_id == store_state.store_id for li in o.items if li.item.lower() == item_lower)
            unallocated = max(0.0, pushed_qty - allocated_qty) if is_pushed else 0.0

            if unallocated > 0:
                unallocated_str = f"<span class='highlight-waste'>{unallocated:.1f}</span>"
            else:
                unallocated_str = f"{unallocated:.1f}" if is_pushed else "0.0"

            rows_inv.append({
                "Item": item.name,
                "Category": category,
                "Full Inv. Qty": f"{item.quantity:.1f}",
                "Unit": item.unit,
                "Days to Expiry": item.days_to_expiry,
                "Pushed?": pushed_str,
                "Pushed Qty": pushed_qty_str,
                "Allocated to Care Home": allocated_str,
                "Unallocated": unallocated_str
            })

        df_inv = pd.DataFrame(rows_inv)
        table_html = df_inv.to_html(classes="data-table", border=0, index=False, escape=False)
        store_tables_html.append(f"""
        <div style="margin-top: 20px; margin-bottom: 30px;">
            <h3 style="color: #EAFBFF; font-size: 15px; border-bottom: 1px solid rgba(4, 216, 217, 0.2); padding-bottom: 5px; margin-bottom: 10px;">{store_name}</h3>
            {table_html}
        </div>
        """)
    section_1_html = "\n".join(store_tables_html)

    # 2. Care Home Funnel Table & Summary
    temp_ledger = StockLedger(sim_day)
    ch_rows = []
    drop_reasons_blocks = []
    sorted_care_homes = sorted(world.care_homes, key=lambda c: c.care_home_id)
    result_map = {r.care_home_id: r for r in negotiation_results}

    for ch in sorted_care_homes:
        result = result_map.get(ch.care_home_id)
        available_count = sum(1 for qty in temp_ledger.get_cross_store_totals().values() if qty > 0)

        all_ledger_items = []
        for store_state in sim_day.stores:
            for item in store_state.pushed_inventory:
                remaining = temp_ledger.get_available(store_state.store_id, item.name)
                if remaining > 0:
                    all_ledger_items.append(DailyFoodItem(
                        name=item.name,
                        days_to_expiry=item.days_to_expiry,
                        quantity=remaining,
                        unit=item.unit
                    ))
        deduped = {}
        for item in all_ledger_items:
            key = item.name.lower()
            if key in deduped:
                existing = deduped[key]
                deduped[key] = DailyFoodItem(
                    name=item.name,
                    days_to_expiry=min(existing.days_to_expiry, item.days_to_expiry),
                    quantity=existing.quantity + item.quantity,
                    unit=item.unit,
                )
            else:
                deduped[key] = item
        eligible_items = hard_constraint_filter(ch, list(deduped.values()), world.catalog)
        eligible_count = len(eligible_items)

        offered_count = len(result.agreed_items) if result else 0
        agreed_count = sum(1 for li in result.agreed_items if li.accepted_quantity > 0) if (result and result.status == "agreed") else 0

        sourcing = {"assignments": [], "deferred": [], "needs_commercial": []}
        if result and result.status == "agreed" and result.agreed_items:
            sourcing = single_store_candidate(
                requested_items=result.agreed_items,
                urgent_item_names=set(result.urgent_item_names),
                ledger=temp_ledger,
                catalog=world.catalog,
            )
            for assignment in sourcing["assignments"]:
                store_id = assignment["store_id"]
                for li in assignment["items"]:
                    temp_ledger.deduct(store_id, li.item, li.accepted_quantity)

        sourced_count = sum(len(assign["items"]) for assign in sourcing["assignments"])
        deferred_count = len(sourcing["deferred"])

        urgent_flagged = result.urgent_item_names if result else []
        urgent_flagged_str = ", ".join(urgent_flagged) if urgent_flagged else "None"
        if urgent_flagged:
            urgent_flagged_str = f"<span class='urgent-highlight'>{urgent_flagged_str}</span>"

        status_text = "Agreed"
        if result is None:
            status_text = "No Result"
        elif result.status == "rejected":
            status_text = "Rejected"
        elif not ch.negotiates_via_a2a:
            status_text = "Auto-Accepted"

        veg_str = "Yes" if ch.hard_constraints.vegetarian_only else "No"

        ch_rows.append({
            "Care Home": ch.name,
            "Residents": ch.resident_count,
            "Veg Only": veg_str,
            "Available Items": available_count,
            "Eligible (post-filter)": eligible_count,
            "Offered": offered_count,
            "Agreed": agreed_count,
            "Sourced": sourced_count,
            "Deferred": deferred_count,
            "Urgent Items Flagged": urgent_flagged_str,
            "Status": status_text
        })

        non_veg_removed = [item.name for item in deduped.values() if ch.hard_constraints.vegetarian_only and item.name.lower() in _NON_VEGETARIAN_ITEMS]
        hard_constraint_removed = [item.name for item in deduped.values() if item.name.lower() in {n.item.lower() for n in ch.memory_notes if n.type == "exclude"}]

        quantity_caps = {note.item.lower(): note.value for note in ch.memory_notes if note.type == "max_quantity" and note.value is not None}
        capped_list = []
        for item in deduped.values():
            if item.name.lower() in quantity_caps:
                cap = quantity_caps[item.name.lower()]
                if item.quantity > cap:
                    capped_list.append(f"{item.name} ({item.quantity:.1f} &rarr; {cap:.1f} {item.unit})")

        offered_names = {oi.item.lower() for oi in result.agreed_items} if result else set()
        not_offered = [it.name for it in eligible_items if it.name.lower() not in offered_names]

        excluded_neg = []
        reduced_neg = []
        if result and result.status == "agreed":
            for li in result.agreed_items:
                if li.accepted_quantity == 0 and li.offered_quantity > 0:
                    excluded_neg.append(li.item)
                elif 0 < li.accepted_quantity < li.offered_quantity:
                    reduced_neg.append(f"{li.item} ({li.offered_quantity:.1f} &rarr; {li.accepted_quantity:.1f} {li.unit})")

        deferred_list = [f"{li.item} ({li.accepted_quantity:.1f} {li.unit})" for li in sourcing["deferred"]]

        drop_lines = [
            f"<b>{ch.name}:</b>",
            f"&nbsp;&nbsp;&bull; Hard constraint filter: {', '.join(non_veg_removed) if non_veg_removed else 'None'} (Non-Veg)" if ch.hard_constraints.vegetarian_only else None,
            f"&nbsp;&nbsp;&bull; Memory note exclusion: {', '.join(hard_constraint_removed) if hard_constraint_removed else 'None'}",
            f"&nbsp;&nbsp;&bull; Memory note cap: {', '.join(capped_list) if capped_list else 'None'}",
            f"&nbsp;&nbsp;&bull; Matchmaker judgment: {', '.join(not_offered) if not_offered else 'None'} (Not offered)",
            f"&nbsp;&nbsp;&bull; Negotiation: {', '.join(excluded_neg) if excluded_neg else 'None'} excluded, {', '.join(reduced_neg) if reduced_neg else 'None'} reduced" if result and result.status == "agreed" else f"&nbsp;&nbsp;&bull; Negotiation: Entire offer rejected" if result and result.status == "rejected" else None,
            f"&nbsp;&nbsp;&bull; Sourcing deferral: {', '.join(deferred_list) if deferred_list else 'None'}"
        ]
        drop_lines = [l for l in drop_lines if l is not None]
        drop_reasons_blocks.append(f"<div style='margin-bottom: 12px; font-size: 13px; line-height: 1.5;'>{'<br/>'.join(drop_lines)}</div>")

    df_ch = pd.DataFrame(ch_rows)
    table_ch_html = df_ch.to_html(classes="data-table care-home-table", border=0, index=False, escape=False)
    table_ch_html = highlight_rejected_rows(table_ch_html, ch_rows)

    section_2_html = f"""
    {table_ch_html}
    <div style="margin-top: 25px; padding: 15px; background: #000F2E; border: 1px solid rgba(4, 216, 217, 0.4); border-radius: 8px;">
        <h3 style="color: #04D8D9; font-size: 14px; margin-top: 0; margin-bottom: 15px; font-family: 'Century Gothic Bold', sans-serif;">Items dropped at each stage:</h3>
        {"".join(drop_reasons_blocks)}
    </div>
    """

    # 3. Volunteer Status Table
    vol_rows = []
    vol_assignments = {}
    for d in deliveries:
        if d.volunteer_id:
            vol_assignments[d.volunteer_id] = d.delivery_id

    sorted_volunteers = sorted(world.volunteers, key=lambda v: v.volunteer_id)
    vol_daily_states = {v.volunteer_id: v.available for v in sim_day.volunteers}

    for vol in sorted_volunteers:
        is_avail = vol_daily_states.get(vol.volunteer_id, False)
        avail_str = "Yes" if is_avail else "No"
        status = "Unavailable Today"
        reason = "—"

        if is_avail:
            if vol.volunteer_id in vol_assignments:
                del_id = vol_assignments[vol.volunteer_id]
                status = f"Assigned — {del_id[:8]}"
            else:
                closest_delivery = None
                min_dist = 999999.0
                for d in deliveries:
                    store = sm.get(d.store_id)
                    if store:
                        dist = approx_distance_minutes(vol.latitude, vol.longitude, store.latitude, store.longitude)
                        if dist < min_dist:
                            min_dist = dist
                            closest_delivery = d
                if closest_delivery:
                    d_orders = [om_lookup[oid] for oid in closest_delivery.order_ids if oid in om_lookup]
                    payload_kg = sum(li.accepted_quantity for o in d_orders for li in o.items)
                    if vol.capacity_kg < payload_kg:
                        status = "Skipped — Insufficient Capacity"
                        reason = f"Payload {payload_kg:.1f}kg exceeds {vol.vehicle_type.replace('_', ' ')} capacity {vol.capacity_kg:.1f}kg"
                    else:
                        ch_coords = []
                        for o in d_orders:
                            ch = chm.get(o.care_home_id)
                            if ch:
                                ch_coords.append([ch.latitude, ch.longitude])
                        store = sm.get(closest_delivery.store_id)
                        route_time = estimate_vol_time(vol, store, ch_coords)
                        if route_time > 120:
                            status = "Skipped — Time Budget Exceeded"
                            reason = f"Route time {route_time:.1f}min exceeds 120min budget"
                        else:
                            status = "Skipped — Not Nearest Available"
                            if closest_delivery.volunteer_id:
                                assigned_vol = vm.get(closest_delivery.volunteer_id)
                                assigned_name = assigned_vol.name if assigned_vol else closest_delivery.volunteer_id
                                reason = f"{assigned_name} was closer and available"
                            else:
                                reason = "Store truck or commercial fallback preferred"
                else:
                    status = "Skipped — Not Nearest Available"
                    reason = "No active deliveries in vicinity today"

        vol_rows.append({
            "Volunteer": vol.name,
            "Vehicle": vol.vehicle_type.replace("_", " ").title(),
            "Capacity (kg)": f"{vol.capacity_kg:.1f}",
            "Available Today": avail_str,
            "Dispatch Status": status,
            "Reason if Skipped": f"<span class='reason-skipped'>{reason}</span>" if reason != "—" else "—",
            "_status_raw": status
        })

    df_vol = pd.DataFrame(vol_rows)
    df_vol_render = df_vol.drop(columns=["_status_raw"])
    table_vol_html = df_vol_render.to_html(classes="data-table volunteer-table", border=0, index=False, escape=False)
    section_3_html = highlight_volunteer_rows(table_vol_html, vol_rows)

    # 4. Route and Delivery Decisions Table
    del_rows = []
    for d in deliveries:
        store = sm.get(d.store_id)
        store_name = store.name if store else d.store_id

        method_str = d.method.replace("_", " ").title()

        if d.method == "volunteer" and d.volunteer_id:
            vol = vm.get(d.volunteer_id)
            assigned_to = vol.name if vol else d.volunteer_id
        elif d.method == "store_truck":
            assigned_to = "Store Truck"
        else:
            assigned_to = "Dunzo (sim)"

        for oid in d.order_ids:
            order = om_lookup.get(oid)
            if not order:
                continue
                
            ch = chm.get(order.care_home_id)
            care_home_str = ch.name if ch else order.care_home_id

            items_summary = get_items_summary(order.items)
            payload_kg = sum(li.accepted_quantity for li in order.items)

            urgent = _urgent_in_delivery(d, om_lookup) # d.method logic needs this? Wait, fallback needs payload_kg.
            # actually we can keep fallback reason same for all rows of this delivery. We need the TOTAL payload for fallback logic.
            total_payload_kg = sum(li.accepted_quantity for ord_obj in [om_lookup.get(id) for id in d.order_ids if om_lookup.get(id)] for li in ord_obj.items)
            
            urgent_in_del = _urgent_in_delivery(d, om_lookup)
            fallback = get_fallback_reason(d, total_payload_kg, bool(urgent_in_del), world, sim_day)

            urgent_names = [
                li.item for li in order.items 
                if li.accepted_quantity > 0 and li.item.lower() in {u.lower() for u in order.urgent_essential_items}
            ]
            urgent_str = ", ".join(f"★ {u}" for u in urgent_names) if urgent_names else "—"

            del_rows.append({
                "Delivery ID": d.delivery_id,
                "Store": store_name,
                "Care Home(s)": care_home_str,
                "Items Summary": items_summary,
                "Total Payload (kg)": f"{payload_kg:.1f}",
                "Method": method_str,
                "Assigned To": assigned_to,
                "Pickup Time": d.pickup_time or "Today 2:00 PM",
                "Urgent Items": urgent_str,
                "Fallback Reason": fallback,
                "_method_raw": d.method
            })

    df_del = pd.DataFrame(del_rows)
    if not df_del.empty:
        df_del_render = df_del.drop(columns=["_method_raw"])
    else:
        df_del_render = pd.DataFrame(columns=["Delivery ID", "Store", "Care Home(s)", "Items Summary", "Total Payload (kg)", "Method", "Assigned To", "Pickup Time", "Urgent Items", "Fallback Reason"])
        
    table_del_html = df_del_render.to_html(classes="data-table delivery-table", border=0, index=False, escape=False)
    section_4_html = highlight_delivery_rows(table_del_html, del_rows)

    delivery_table, delivery_html = generate_delivery_table(deliveries, orders, world)
    map_filepath, map_html = generate_map(deliveries, orders, world, run_id)
    negotiation_report = generate_negotiation_report(negotiation_results, orders, world)
    audit_report = generate_audit_report(sim_day, world, dispatch_stats)
    summary_filepath, summary_html = generate_summary_page(deliveries, orders, negotiation_results, dispatch_stats, world, sim_day, run_id)
    message_log = get_message_log()

    use_api_links = os.environ.get("RUNNING_ON_CLOUD_RUN") or os.environ.get("SERVED_VIA_API")
    map_link = "/map" if use_api_links else f"map_{run_id}.html"
    summary_link = "/" if use_api_links else "latest_summary.html"

    # Create the fully styled HTML Page
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SurplusCart - Daily Operations Report</title>
  <style>
    body {{
      background-color: #000517;
      color: #EAFBFF;
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }}
    
    header, footer {{
      width: 100%%;
      max-width: 1600px;
      margin: 0 auto;
    }}
    header img, footer img {{
      width: 100%%;
      height: auto;
      display: block;
    }}

    .title-block {{
      text-align: center;
      margin: 20px auto 40px;
    }}
    .title-block h1 {{
      font-family: 'Century Gothic Bold', 'Arial Bold', sans-serif;
      color: #04D8D9;
      font-size: 28px;
      margin: 0 0 10px 0;
      text-shadow: 0 0 8px #01F3F4, 0 0 16px #04D8D9;
    }}
    .title-block p {{
      font-size: 14px;
      color: #EAFBFF;
      margin: 0;
    }}

    .content-container {{
      width: 90%%;
      max-width: 1200px;
      margin: 0 auto;
      flex-grow: 1;
    }}

    .card {{
      background-color: #000F2E;
      border: 1px solid rgba(4, 216, 217, 0.4);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 30px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }}
    
    .card h2 {{
      font-family: 'Century Gothic Bold', 'Arial Bold', sans-serif;
      color: #04D8D9;
      font-size: 18px;
      margin-top: 0;
      margin-bottom: 20px;
      text-shadow: 0 0 8px #01F3F4, 0 0 16px #04D8D9;
    }}

    .data-table {{
      width: 100%%;
      border-collapse: collapse;
      margin-bottom: 20px;
      font-size: 14px;
      color: #EAFBFF;
    }}
    .data-table th {{
      background-color: #087C81;
      color: #EAFBFF;
      text-align: left;
      padding: 10px 12px;
      font-weight: 600;
    }}
    .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(4, 216, 217, 0.4);
    }}
    .data-table tr:nth-child(even) {{
      background-color: #000F2E;
    }}
    .data-table tr:nth-child(odd) {{
      background-color: #001433;
    }}
    
    .highlight-waste {{
      color: #01F3F4;
      font-weight: bold;
    }}
    .rejected-row {{
      background-color: #087C81 !important;
      color: #EAFBFF;
    }}
    .urgent-highlight {{
      color: #01F3F4;
      font-weight: bold;
    }}
    
    .reason-skipped {{
      color: #087C81;
    }}


    .btn-map {{
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 1000;
      background-color: #04D8D9;
      color: #000517;
      padding: 12px 32px;
      font-family: 'Century Gothic Bold', sans-serif;
      font-size: 16px;
      font-weight: bold;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
      transition: all 0.3s ease;
      box-shadow: 0 0 8px rgba(4, 216, 217, 0.4);
    }}
    .btn-map:hover {{
      background-color: #01F3F4;
      box-shadow: 0 0 12px #01F3F4;
    }}
  </style>
</head>
<body>
  <div style="position:fixed; top:20px; right:20px; z-index:1000; display:flex; gap:10px;">
    <a href="{summary_link}" target="_blank" class="btn-top">View Summary</a>
    <a href="{map_link}" target="_blank" class="btn-top">View Map</a>
  </div>
  
  <header>
    <img src="../assets/header.png" alt="Header Image" onerror="this.style.display='none';"/>
  </header>

  <div class="title-block">
    <h1>SurplusCart &mdash; Agentic Food Rescue</h1>
    <p>Daily Operations Report | Run ID: {run_id}</p>
  </div>

  <div class="content-container">
    <!-- SECTION 1 -->
    <div class="card" id="section-1">
      <h2>Store Inventory Report</h2>
      {section_1_html}
    </div>

    <!-- SECTION 2 -->
    <div class="card" id="section-2">
      <h2>Care Home Allocation Summary</h2>
      {section_2_html}
      <h3 style="color: #04D8D9; font-size: 14px; margin-top: 30px; margin-bottom: 10px; font-family: 'Century Gothic Bold', sans-serif;">Negotiation Transcripts</h3>
      {negotiation_report}
    </div>

    <!-- SECTION 3 -->
    <div class="card" id="section-3">
      <h2>Volunteer Availability and Dispatch Status</h2>
      {section_3_html}
    </div>

    <!-- SECTION 4 -->
    <div class="card" id="section-4">
      <h2>Delivery Routes and Dispatch Decisions</h2>
      {section_4_html}
      {audit_report}
    </div>
  </div>

  <footer>
    <img src="../assets/footer.png" alt="Footer Image" onerror="this.style.display='none';"/>
  </footer>
</body>
</html>
"""

    report_filepath = str(_REPORTS_DIR / f"report_{run_id}.html")
    latest_filepath = str(_REPORTS_DIR / "latest_report.html")

    if not os.environ.get("RUNNING_ON_CLOUD_RUN"):
        with open(report_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        with open(latest_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

    return {
        "report_html": html_content,
        "map_html": map_html,
        "summary_html": summary_html,
        "report_filepath": report_filepath,
        "map_filepath": map_filepath,
        "summary_filepath": summary_filepath,
        "delivery_table": delivery_table,
        "negotiation_report": negotiation_report,
        "audit_report": audit_report,
        "message_log": message_log,
        "stats": dispatch_stats
    }
