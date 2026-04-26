import { useEffect, useState } from "react";
export function Database() {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<"monthly" | "daily">("monthly");
  const [product, setProduct] = useState("shrimp");

  useEffect(() => {
    setLoading(true);
    setData([]);

const url =
  mode === "monthly"
    ? `/api/raw?product=${product}`
    : `/api/raw-daily`;

    console.log("FETCHING:", url);

    fetch(url, { cache: "no-store" })
      .then(res => res.json())
      .then(res => {
        console.log("DATA:", product, res.slice(0, 3));
        setData(res);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });

  }, [mode, product]);

  if (loading) return <div className="p-6">Loading...</div>;
  if (data.length === 0) return <div className="p-6">No data</div>;

  const columns = Object.keys(data[0]);

  return (
      <div className="p-6 space-y-4">
        <h1 className="text-xl">
          Database
          ({mode === "daily" ? "Shrimp Daily" : `${product.charAt(0).toUpperCase() + product.slice(1)} Monthly`})
        </h1>
        <div className="flex gap-2 items-center">

          <button
              onClick={() => setMode("monthly")}
              className={`px-4 py-2 rounded ${
                  mode === "monthly" ? "bg-blue-600 text-white" : "bg-gray-200"
              }`}
          >
            Monthly
          </button>

          <button
              onClick={() => setMode("daily")}
              className={`px-4 py-2 rounded ${
                  mode === "daily" ? "bg-blue-600 text-white" : "bg-gray-200"
              }`}
          >
            Daily
          </button>

          {mode === "monthly" && (
              <select
                  value={product}
                  onChange={(e) => setProduct(e.target.value)}
                  className="px-3 py-2 border rounded"
              >
                <option value="shrimp">Shrimp</option>
                <option value="salmon">Salmon</option>
                <option value="tuna">Tuna</option>
                <option value="whitefish">Whitefish</option>
              </select>
          )}

        </div>

        <div className="overflow-auto border rounded">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-100">
            <tr>
              {columns.map(col => (
                  <th key={col} className="p-2 border">
                    {col}
                  </th>
              ))}
            </tr>
            </thead>

            <tbody>
            {data.map((row, i) => (
                <tr key={i}>
                  {columns.map(col => (
                      <td key={col} className="p-2 border text-center">
                        {typeof row[col] === "number"
                            ? row[col].toLocaleString()
                            : row[col] ?? "-"}
                      </td>
                  ))}
                </tr>
            ))}
            </tbody>
          </table>
        </div>
      </div>
  );
}